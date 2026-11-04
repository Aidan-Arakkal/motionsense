from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from .model import TemporalCNN


class Predictor:
    def __init__(self, artifact_dir: Path):
        torch.set_num_threads(4)
        bundle = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        if bundle["schema_version"] != 1:
            raise ValueError("Unsupported model schema")
        self.net = TemporalCNN()
        self.net.load_state_dict(bundle["state_dict"])
        self.net.eval()
        self.mean, self.std = bundle["mean"].numpy(), bundle["std"].numpy()
        self.temperature = bundle["temperature"]
        self.labels, self.channels = bundle["labels"], bundle["channels"]

    def predict(self, window):
        x = np.asarray(window, dtype=np.float32)
        if x.shape != (128, 9) or not np.isfinite(x).all():
            raise ValueError("window must contain 128 time steps by 9 finite sensor values")
        x = (x.T[None] - self.mean) / self.std
        if not np.isfinite(x).all():
            raise ValueError("Sensor values overflow after normalization")
        with torch.inference_mode():
            logits = self.net(torch.from_numpy(x)).numpy()[0]
        probabilities = softmax(logits / self.temperature)
        idx = int(probabilities.argmax())
        return {"prediction": self.labels[idx], "confidence": float(probabilities[idx]), "status": "accepted" if probabilities[idx] >= .8 else "review", "probabilities": {label: float(p) for label, p in zip(self.labels, probabilities)}, "model_version": "1.0.0"}

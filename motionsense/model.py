import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 9
        for width in (32, 64, 96):
            layers += [nn.Conv1d(channels, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)]
            channels = width
        self.network = nn.Sequential(*layers, nn.Flatten(), nn.Dropout(0.35), nn.Linear(96*16, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, 6))

    def forward(self, x):
        return self.network(x)


def fit_temperature(logits, labels):
    def loss(t):
        z = logits / t
        return float((logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]).mean())
    return float(minimize_scalar(loss, bounds=(0.2, 5.0), method="bounded").x)


def calibration_metrics(logits, labels, temperature=1):
    p = softmax(logits / temperature, axis=1)
    confidence, pred = p.max(1), p.argmax(1)
    ece = 0.0
    for i in range(15):
        mask = (confidence > i/15) & (confidence <= (i+1)/15)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == labels[mask]).mean() - confidence[mask].mean())
    z = logits / temperature
    return {"nll": float((logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]).mean()), "ece_15_bins": float(ece)}

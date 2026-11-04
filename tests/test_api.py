import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from motionsense.api import app, ARTIFACTS
from motionsense.inference import Predictor


@pytest.fixture(scope="module")
def client():
    if not (ARTIFACTS / "model.pt").exists():
        pytest.skip("Run python -m motionsense.train to enable artifact integration tests")
    with TestClient(app) as client:
        yield client


def test_served_predictions_match_checkpoint_and_are_normalized(client):
    examples = client.get("/examples").json()
    predictor = Predictor(ARTIFACTS)
    for example in examples:
        response = client.post("/predict", json={"window": example["window"]})
        assert response.status_code == 200
        served = response.json()
        direct = predictor.predict(example["window"])
        assert served == direct
        assert sum(served["probabilities"].values()) == pytest.approx(1, abs=1e-6)
        assert served["confidence"] == max(served["probabilities"].values())
        assert served["status"] == ("accepted" if served["confidence"] >= .8 else "review")


@pytest.mark.parametrize("body", [{"window": [[0]*9]*127}, {"window": [[0]*8]*128}, {"window": [[0]*9]*128, "unexpected": True}, {"window": [["invalid"]*9]*128}, {"window": [["NaN"]*9]*128}])
def test_api_rejects_invalid_inputs(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_artifact_splits_and_routes(client):
    metrics = client.get("/metrics").json()
    groups = [set(v) for v in metrics["split_subjects"].values()]
    assert all(not a & b for i,a in enumerate(groups) for b in groups[i+1:])
    assert client.get("/health").json()["status"] == "ready"
    assert client.get("/schema").json()["shape"] == [128,9]
    assert "Signal explorer" in client.get("/").text

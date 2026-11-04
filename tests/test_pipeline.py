import numpy as np
import pytest
import torch
from motionsense.data import fit_scaler, split_subjects
from motionsense.model import TemporalCNN, calibration_metrics, fit_temperature
from motionsense.inference import Predictor


def test_overlapping_subject_windows_never_cross_partitions():
    subjects = np.repeat(np.arange(1, 22), 40)
    split = split_subjects(subjects)
    sets = [set(subjects[index]) for index in split.values()]
    assert len(set.union(*sets)) == 21
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert sum(map(len, split.values())) == len(subjects)


def test_scaler_uses_only_supplied_training_data():
    train = np.arange(18*128, dtype=np.float32).reshape(2, 9, 128)
    mean, std = fit_scaler(train)
    transformed = (train - mean) / std
    np.testing.assert_allclose(transformed.mean(axis=(0, 2)), 0, atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=(0, 2)), 1, atol=1e-6)
    assert np.isfinite((np.zeros((2,9,128)) - mean) / std).all()
    assert np.all(fit_scaler(np.zeros((2,9,128)))[1] > 0)


def test_network_is_sensitive_to_signal_and_supports_backprop():
    torch.manual_seed(3)
    net = TemporalCNN()
    x = torch.randn(4, 9, 128, requires_grad=True)
    output = net(x)
    assert output.shape == (4, 6)
    torch.nn.functional.cross_entropy(output, torch.tensor([0,1,2,3])).backward()
    assert x.grad.abs().sum() > 0


def test_temperature_fit_reduces_overconfident_log_loss():
    logits = np.array([[9.,0.],[9.,0.],[9.,0.],[9.,0.]])
    labels = np.array([0,0,0,1])
    t = fit_temperature(logits, labels)
    assert calibration_metrics(logits, labels, t)["nll"] < calibration_metrics(logits, labels)["nll"]


@pytest.mark.parametrize("window", [np.zeros((127,9)), np.zeros((128,8)), np.full((128,9), np.nan), np.full((128,9), np.inf)])
def test_inference_rejects_bad_shape_or_values(window):
    predictor = Predictor.__new__(Predictor)
    with pytest.raises(ValueError, match="128 time steps"):
        predictor.predict(window)

import torch
import pytest
from kalman.filters import KalmanFilter
from kalman.gaussian import GaussianState

@pytest.fixture
def dummy_kf():
    F = torch.eye(2)            # Process matrix
    H = torch.eye(2)            # Measurement matrix
    Q = 0.01 * torch.eye(2)     # Process noise
    R = 0.1 * torch.eye(2)      # Measurement noise
    return KalmanFilter(F, H, Q, R)

@pytest.fixture
def initial_state():
    mean = torch.tensor([1.0, 2.0])
    cov = torch.eye(2)
    return GaussianState(mean, cov)

def test_predict(dummy_kf, initial_state):
    pred = dummy_kf.predict(initial_state)
    assert torch.allclose(pred.mean, initial_state.mean)  # since F = I
    assert pred.covariance.shape == torch.Size([2, 2])
    assert not torch.equal(pred.covariance, initial_state.covariance)

def test_project(dummy_kf, initial_state):
    projected = dummy_kf.project(initial_state)
    assert projected.mean.shape == torch.Size([2])
    assert projected.covariance.shape == torch.Size([2, 2])
    assert projected.precision is not None
    assert torch.allclose(projected.precision @ projected.covariance.mT, torch.eye(2), atol=1e-3)

def test_update(dummy_kf, initial_state):
    measurement = torch.tensor([1.5, 2.5])
    updated = dummy_kf.update(initial_state, measurement)
    assert updated.mean.shape == torch.Size([2])
    assert updated.covariance.shape == torch.Size([2, 2])

def test_predict_update_cycle(dummy_kf, initial_state):
    measurement = torch.tensor([1.2, 2.1])
    updated = dummy_kf.update(initial_state, measurement)
    predicted = dummy_kf.predict(updated)
    assert predicted.mean.shape == torch.Size([2])
    assert predicted.covariance.shape == torch.Size([2, 2])


@pytest.fixture
def batched_kf():
    batch_size = 4
    state_dim = 2
    obs_dim = 2
    F = torch.eye(state_dim).repeat(batch_size, 1, 1)  # (B, state_dim, state_dim)
    H = torch.eye(obs_dim).repeat(batch_size, 1, 1)    # (B, obs_dim, state_dim)
    Q = 0.01 * torch.eye(state_dim).repeat(batch_size, 1, 1)
    R = 0.1 * torch.eye(obs_dim).repeat(batch_size, 1, 1)
    return KalmanFilter(F, H, Q, R)

@pytest.fixture
def batched_state():
    batch_size = 4
    state_dim = 2
    mean = torch.rand(batch_size, state_dim)
    cov = torch.stack([torch.eye(state_dim) for _ in range(batch_size)])  # (B, D, D)
    return GaussianState(mean, cov)

def test_batched_predict(batched_kf, batched_state):
    pred = batched_kf.predict(batched_state)
    assert pred.mean.shape == (4, 2)
    assert pred.covariance.shape == (4, 2, 2)

def test_batched_project(batched_kf, batched_state):
    projected = batched_kf.project(batched_state)
    assert projected.mean.shape == (4, 2)
    assert projected.covariance.shape == (4, 2, 2)
    assert projected.precision is not None
    # Test that precision is roughly the inverse of covariance
    eye = torch.eye(2)
    for p, c in zip(projected.precision, projected.covariance):
        approx = p @ c.mT
        assert torch.allclose(approx, eye, atol=1e-3)

def test_batched_update(batched_kf, batched_state):
    measurement = torch.rand(4, 2)
    updated = batched_kf.update(batched_state, measurement)
    assert updated.mean.shape == (4, 2)
    assert updated.covariance.shape == (4, 2, 2)

def test_batched_predict_update_cycle(batched_kf, batched_state):
    measurement = torch.rand(4, 2)
    updated = batched_kf.update(batched_state, measurement)
    predicted = batched_kf.predict(updated)
    assert predicted.mean.shape == (4, 2)
    assert predicted.covariance.shape == (4, 2, 2)
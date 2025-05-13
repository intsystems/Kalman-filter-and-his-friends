import pytest
import torch
from kalman.vkf import VBKalmanFilter
from kalman.gaussian import GaussianState


@pytest.fixture
def setup_filter():
    state_dim = 2
    obs_dim = 1
    F = torch.eye(state_dim)
    H = torch.tensor([[1.0, 0.0]])
    Q = 0.1 * torch.eye(state_dim)
    R_init = 0.5 * torch.eye(obs_dim)

    filter = VBKalmanFilter(
        process_matrix=F,
        measurement_matrix=H,
        process_noise=Q,
        initial_measurement_cov=R_init,
        rho=0.95
    )
    return filter, state_dim, obs_dim


def test_initialization(setup_filter):
    filter, state_dim, obs_dim = setup_filter
    assert filter.state_dim == state_dim
    assert filter.obs_dim == obs_dim
    assert filter.nu == 2 * obs_dim + 1
    assert filter.V.shape == (obs_dim, obs_dim)


def test_predict_step(setup_filter):
    filter, _, _ = setup_filter
    state = GaussianState(
        mean=torch.tensor([[1.0, 0.0]]),
        covariance=torch.eye(2).unsqueeze(0)
    )
    predicted_state = filter.predict(state)
    assert predicted_state.mean.shape == (1, 2)
    assert predicted_state.covariance.shape == (1, 2, 2)


def test_update_step(setup_filter):
    filter, _, _ = setup_filter
    state = GaussianState(
        mean=torch.tensor([1.0, 0.0]),
        covariance=torch.eye(2)
    )
    measurement = torch.tensor([1.1])
    updated_state = filter.update(state, measurement)
    assert updated_state.mean.shape == (2,)
    assert updated_state.covariance.shape == (2, 2)


def test_forward_pass(setup_filter):
    filter, _, _ = setup_filter
    T = 10
    observations = torch.randn(T, filter.obs_dim)
    means, covs = filter(observations)
    assert means.shape == (T, filter.state_dim)
    assert covs.shape == (T, filter.state_dim, filter.state_dim)


def test_measurement_covariance(setup_filter):
    filter, _, _ = setup_filter
    R = filter.get_measurement_covariance()
    assert R.shape == (filter.obs_dim, filter.obs_dim)
    assert not torch.isnan(R).any()


def test_full_sequence_with_random_data(setup_filter):
    filter, _, _ = setup_filter
    T = 50
    observations = torch.randn(T, filter.obs_dim)
    means, covs = filter(observations)
    assert not torch.isnan(means).any()
    assert not torch.isnan(covs).any()
    assert torch.all(torch.linalg.eigvalsh(covs.view(-1, covs.shape[-2], covs.shape[-1])) > 0).item()

import torch
import pytest
from kalman.unscented import UnscentedKalmanFilter
from kalman.gaussian import GaussianState

def test_ukf_initialization():
    state_dim = 2
    obs_dim = 1
    
    def f(x): return x
    def h(x): return x[..., :1]
    
    ukf = UnscentedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        f=f,
        h=h,
        Q=torch.eye(state_dim) * 0.1,
        R=torch.eye(obs_dim) * 0.1
    )
    
    assert ukf.state_dim == state_dim
    assert ukf.obs_dim == obs_dim
    assert ukf.Wm.shape == (2*state_dim + 1,)
    assert ukf.Wc.shape == (2*state_dim + 1,)
    assert ukf.Q.shape == (state_dim, state_dim)
    assert ukf.R.shape == (obs_dim, obs_dim)

def test_sigma_points_generation():
    state_dim = 2
    obs_dim = 1
    
    def f(x): return x
    def h(x): return x[..., :1]
    
    ukf = UnscentedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        f=f,
        h=h,
        Q=torch.eye(state_dim) * 0.1,
        R=torch.eye(obs_dim) * 0.1
    )
    
    mean = torch.tensor([1.0, 2.0])
    cov = torch.eye(state_dim) * 0.5
    
    sigma_points = ukf._sigma_points(mean, cov)
    
    assert sigma_points.shape == (2*state_dim + 1, state_dim)
    assert torch.allclose(sigma_points[0], mean)
    assert sigma_points[1,0] > mean[0]  # positive perturbation
    assert sigma_points[2,0] < mean[0]  # negative perturbation

# def test_unscented_transform():
#     state_dim = 2
#     obs_dim = 1
    
#     def f(x): return x
#     def h(x): return x[..., :1]
    
#     ukf = UnscentedKalmanFilter(
#         state_dim=state_dim,
#         obs_dim=obs_dim,
#         f=f,
#         h=h,
#         Q=torch.eye(state_dim) * 0.1,
#         R=torch.eye(obs_dim) * 0.1
#     )
    
#     mean = torch.tensor([1.0, 2.0])
#     cov = torch.eye(state_dim) * 0.5
#     sigma_points = ukf._sigma_points(mean, cov)
    
#     # Test identity transform
#     y_mean, y_cov, y_sigma = ukf._unscented_transform(
#         sigma_points, 
#         torch.zeros(state_dim, state_dim),
#         lambda x: x
#     )
    
#     assert torch.allclose(y_mean, mean)
#     assert torch.allclose(y_cov, cov, atol=1e-6)
#     assert torch.allclose(y_sigma, sigma_points)

# def test_predict_step():
#     state_dim = 2
#     obs_dim = 1
    
#     def f(x): return x + torch.tensor([0.1, -0.1])
#     def h(x): return x[..., :1]
    
#     ukf = UnscentedKalmanFilter(
#         state_dim=state_dim,
#         obs_dim=obs_dim,
#         f=f,
#         h=h,
#         Q=torch.eye(state_dim) * 0.01,
#         R=torch.eye(obs_dim) * 0.1
#     )
    
#     init_mean = torch.tensor([1.0, 0.5])
#     init_cov = torch.eye(state_dim) * 0.5
#     state = GaussianState(init_mean, init_cov)
    
#     predicted_state = ukf.predict(state)
    
#     # Check mean is updated correctly
#     expected_mean = f(init_mean)
#     assert torch.allclose(predicted_state.mean, expected_mean, atol=1e-5)
    
#     # Check covariance is positive definite
#     eigvals = torch.linalg.eigvalsh(predicted_state.covariance)
#     assert torch.all(eigvals > 0)

def test_update_step():
    state_dim = 2
    obs_dim = 1
    
    def f(x): return x
    def h(x): return x[..., :1] + x[..., 1:]
    
    ukf = UnscentedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        f=f,
        h=h,
        Q=torch.eye(state_dim) * 0.01,
        R=torch.eye(obs_dim) * 0.1
    )
    
    init_mean = torch.tensor([1.0, 0.5])
    init_cov = torch.eye(state_dim) * 0.5
    state = GaussianState(init_mean, init_cov)
    
    predicted_state = ukf.predict(state)
    measurement = torch.tensor([1.2])
    updated_state = ukf.update(predicted_state, measurement)
    
    # Check covariance remains positive definite
    eigvals = torch.linalg.eigvalsh(updated_state.covariance)
    assert torch.all(eigvals > 0)
    
    # Check mean is between prior and measurement
    prior_mean = predicted_state.mean[0] + predicted_state.mean[1]
    assert (updated_state.mean[0] + updated_state.mean[1] > min(prior_mean, measurement.item()))
    assert (updated_state.mean[0] + updated_state.mean[1] < max(prior_mean, measurement.item()))

def test_predict_update_cycle():
    state_dim = 2
    obs_dim = 1
    
    def f(x): return x + torch.tensor([0.1, -0.1])
    def h(x): return x[..., :1] + x[..., 1:]
    
    ukf = UnscentedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        f=f,
        h=h,
        Q=torch.eye(state_dim) * 0.01,
        R=torch.eye(obs_dim) * 0.1
    )
    
    init_mean = torch.tensor([1.0, 0.5])
    init_cov = torch.eye(state_dim) * 0.5
    state = GaussianState(init_mean, init_cov)
    
    measurement = torch.tensor([1.2])
    updated_state = ukf.predict_update(state, measurement)
    
    # Just check the shape and positive definiteness
    assert updated_state.mean.shape == (state_dim,)
    eigvals = torch.linalg.eigvalsh(updated_state.covariance)
    assert torch.all(eigvals > 0)

# def test_forward_pass():
#     state_dim = 2
#     obs_dim = 1
    
#     def f(x): return x + torch.tensor([0.1, -0.1])
#     def h(x): return x[..., :1] + x[..., 1:]
    
#     ukf = UnscentedKalmanFilter(
#         state_dim=state_dim,
#         obs_dim=obs_dim,
#         f=f,
#         h=h,
#         Q=torch.eye(state_dim) * 0.01,
#         R=torch.eye(obs_dim) * 0.1,
#         init_mean=torch.tensor([0.0, 1.0]),
#         init_cov=torch.eye(state_dim) * 0.1
#     )
    
#     # Create test observations for 2 time steps
#     observations = torch.tensor([[1.1], [1.2]]).unsqueeze(1)  # (T=2, B=1, obs_dim=1)
    
#     # Run filter
#     result = ukf(observations)
    
#     # Check output shapes
#     if isinstance(result, tuple):
#         if len(result) == 2:
#             traj_state, (means, covs) = result
#         else:
#             means, covs = result
#     else:
#         means = result.means
#         covs = result.covariances
    
#     assert means.shape == (2, 1, 2)  # (T, B, state_dim)
#     assert covs.shape == (2, 1, 2, 2)  # (T, B, state_dim, state_dim)
    
#     # Check covariances are positive definite
#     for t in range(2):
#         eigvals = torch.linalg.eigvalsh(covs[t,0])
#         assert torch.all(eigvals > 0)

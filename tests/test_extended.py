import torch
import pytest
from kalman.extended import ExtendedKalmanFilter
from kalman.gaussian import GaussianState

def test_initialization():
    state_dim = 2
    obs_dim = 1
    
    # Simple linear functions for testing
    def f(x): return x
    def h(x): return x.sum(dim=-1, keepdim=True)
    
    # Test default initialization
    ekf = ExtendedKalmanFilter(state_dim, obs_dim, f, h)
    
    assert ekf.state_dim == state_dim
    assert ekf.obs_dim == obs_dim
    assert torch.allclose(ekf.Q, torch.eye(state_dim))
    assert torch.allclose(ekf.R, torch.eye(obs_dim))
    assert torch.allclose(ekf._init_mean, torch.zeros(state_dim))
    assert torch.allclose(ekf._init_cov, torch.eye(state_dim))
    
    # Test custom initialization
    custom_Q = torch.eye(state_dim) * 0.1
    custom_R = torch.eye(obs_dim) * 0.5
    custom_mean = torch.ones(state_dim)
    custom_cov = torch.eye(state_dim) * 2
    
    ekf_custom = ExtendedKalmanFilter(
        state_dim, obs_dim, f, h,
        Q=custom_Q, R=custom_R,
        init_mean=custom_mean, init_cov=custom_cov
    )
    
    assert torch.allclose(ekf_custom.Q, custom_Q)
    assert torch.allclose(ekf_custom.R, custom_R)
    assert torch.allclose(ekf_custom._init_mean, custom_mean)
    assert torch.allclose(ekf_custom._init_cov, custom_cov)

def test_jacobian_computation():
    state_dim = 2
    obs_dim = 2
    
    # Define a simple non-linear system
    def f(x): return torch.stack([x[...,0]**2, x[...,1]], dim=-1)
    def h(x): return torch.stack([x[...,0], x[...,0] * x[...,1]], dim=-1)
    
    # Test with autograd
    ekf = ExtendedKalmanFilter(state_dim, obs_dim, f, h)
    
    test_point = torch.tensor([2.0, 3.0])
    
    # Expected Jacobians
    expected_F = torch.tensor([[4.0, 0.0], [0.0, 1.0]])  # df/dx at x=[2,3]
    expected_H = torch.tensor([[1.0, 0.0], [3.0, 2.0]])  # dh/dx at x=[2,3]
    
    computed_F = ekf._F(test_point)
    computed_H = ekf._H(test_point)
    
    assert torch.allclose(computed_F, expected_F, atol=1e-5)
    assert torch.allclose(computed_H, expected_H, atol=1e-5)
    
    # # Test with provided Jacobians
    # def F_jac(x): return torch.stack([
    #     torch.stack([2*x[...,0], torch.zeros_like(x[...,0])], dim=-1),
    #     torch.stack([torch.zeros_like(x[...,1]), torch.ones_like(x[...,1])], dim=-1)
    # ], dim=-1)
    
    # def H_jac(x): return torch.stack([
    #     torch.stack([torch.ones_like(x[...,0]), torch.zeros_like(x[...,0])], dim=-1),
    #     torch.stack([x[...,1], x[...,0]], dim=-1)
    # ], dim=-1)
    
    # ekf_provided = ExtendedKalmanFilter(
    #     state_dim, obs_dim, f, h,
    #     F_jacobian=F_jac, H_jacobian=H_jac
    # )
    
    # computed_F_provided = ekf_provided._F(test_point)
    # computed_H_provided = ekf_provided._H(test_point)
    
    # assert torch.allclose(computed_F_provided, expected_F, atol=1e-5)
    # assert torch.allclose(computed_H_provided, expected_H, atol=1e-5)

def test_predict_update_linear_system():
    # Test with linear system (should match standard KF)
    state_dim = 2
    obs_dim = 1
    
    F_matrix = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
    H_matrix = torch.tensor([[1.0, 0.0]])
    
    def f(x): 
        return (F_matrix @ x.unsqueeze(-1)).squeeze(-1)
    
    def h(x): 
        return (H_matrix @ x.unsqueeze(-1)).squeeze(-1)
    
    Q = torch.eye(state_dim) * 0.1
    R = torch.eye(obs_dim) * 0.5
    
    # Создаем EKF без явных Jacobian-функций
    ekf = ExtendedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        f=f,
        h=h,
        Q=Q,
        R=R
    )
    
    # Initial state
    init_mean = torch.tensor([1.0, 0.5])
    init_cov = torch.eye(state_dim) * 0.5
    state = GaussianState(init_mean, init_cov)
    
    # Test predict step
    predicted_state = ekf.predict(state)
    
    # Expected prediction
    expected_mean = f(init_mean)
    expected_cov = F_matrix @ init_cov @ F_matrix.T + Q
    
    assert torch.allclose(predicted_state.mean, expected_mean, atol=1e-5)
    assert torch.allclose(predicted_state.covariance, expected_cov, atol=1e-5)
    
    # Test update step
    measurement = torch.tensor([1.2])
    updated_state = ekf.update(predicted_state, measurement)
    
    # Expected update
    H = H_matrix  # Используем постоянную матрицу H
    y = measurement - h(predicted_state.mean)
    S = H @ predicted_state.covariance @ H.T + R
    K = predicted_state.covariance @ H.T @ torch.linalg.inv(S)
    
    expected_upd_mean = predicted_state.mean + (K @ y.unsqueeze(-1)).squeeze(-1)
    I = torch.eye(state_dim)
    ImKH = I - K @ H
    expected_upd_cov = ImKH @ predicted_state.covariance @ ImKH.T + K @ R @ K.T
    
    assert torch.allclose(updated_state.mean, expected_upd_mean, atol=1e-5)
    assert torch.allclose(updated_state.covariance, expected_upd_cov, atol=1e-5)

def test_predict_update_nonlinear_system():
    # Test with non-linear system
    state_dim = 1
    obs_dim = 1
    
    def f(x): 
        return x + 0.1 * torch.sin(x)
    
    def h(x): 
        return x**2
    
    Q = torch.eye(state_dim) * 0.01
    R = torch.eye(obs_dim) * 0.1
    
    # Создаем EKF без явных Jacobian-функций
    ekf = ExtendedKalmanFilter(
        state_dim=state_dim,
        obs_dim=obs_dim,
        f=f,
        h=h,
        Q=Q,
        R=R
    )
    
    # Initial state
    init_mean = torch.tensor([1.0])
    init_cov = torch.eye(state_dim) * 0.5
    state = GaussianState(init_mean, init_cov)
    
    # Test predict step
    predicted_state = ekf.predict(state)
    
    # Expected prediction (вычисляем якобиан аналитически для проверки)
    expected_mean = f(init_mean)
    F = (1 + 0.1 * torch.cos(init_mean)).reshape(1, 1)  # Аналитический якобиан
    expected_cov = F @ init_cov @ F.T + Q
    
    assert torch.allclose(predicted_state.mean, expected_mean, atol=1e-5)
    assert torch.allclose(predicted_state.covariance, expected_cov, atol=1e-5)
    
    # Test update step
    measurement = torch.tensor([1.1])
    updated_state = ekf.update(predicted_state, measurement)
    
    # Expected update (вычисляем якобиан аналитически для проверки)
    H = (2 * predicted_state.mean).reshape(1, 1)  # Аналитический якобиан
    y = measurement - h(predicted_state.mean)
    S = H @ predicted_state.covariance @ H.T + R
    K = predicted_state.covariance @ H.T @ torch.linalg.inv(S)
    
    expected_upd_mean = predicted_state.mean + (K @ y.unsqueeze(-1)).squeeze(-1)
    I = torch.eye(state_dim)
    ImKH = I - K @ H
    expected_upd_cov = ImKH @ predicted_state.covariance @ ImKH.T + K @ R @ K.T
    
    assert torch.allclose(updated_state.mean, expected_upd_mean, atol=1e-5)
    assert torch.allclose(updated_state.covariance, expected_upd_cov, atol=1e-5)

def test_predict_update_combined():
    state_dim = 1
    obs_dim = 1
    
    def f(x): return 1.1 * x
    def h(x): return 0.9 * x
    
    Q = torch.eye(state_dim) * 0.01
    R = torch.eye(obs_dim) * 0.1
    
    ekf = ExtendedKalmanFilter(
        state_dim, obs_dim, f, h,
        Q=Q, R=R
    )
    
    # Initial state
    init_mean = torch.tensor([1.0])
    init_cov = torch.eye(state_dim) * 0.5
    state = GaussianState(init_mean, init_cov)
    
    measurement = torch.tensor([0.95])
    
    # Test combined predict-update
    updated_state = ekf.predict_update(state, measurement)
    
    # Compare with separate steps
    predicted_state = ekf.predict(state)
    updated_state_separate = ekf.update(predicted_state, measurement)
    
    assert torch.allclose(updated_state.mean, updated_state_separate.mean, atol=1e-5)
    assert torch.allclose(updated_state.covariance, updated_state_separate.covariance, atol=1e-5)
    

def test_numerical_stability():
    state_dim = 1
    obs_dim = 1
    
    def f(x): return x
    def h(x): return x
    
    # Very small noise covariances
    Q = torch.eye(state_dim) * 1e-6
    R = torch.eye(obs_dim) * 1e-6
    
    ekf = ExtendedKalmanFilter(
        state_dim, obs_dim, f, h,
        Q=Q, R=R,
        eps=1e-6
    )
    
    # Initial state with very small covariance
    init_mean = torch.tensor([1.0])
    init_cov = torch.eye(state_dim) * 1e-6
    state = GaussianState(init_mean, init_cov)
    
    measurement = torch.tensor([1.0])
    
    # Should not raise numerical errors
    updated_state = ekf.predict_update(state, measurement)
    
    assert torch.isfinite(updated_state.mean).all()
    assert torch.isfinite(updated_state.covariance).all()
    assert (updated_state.covariance > 0).all()  # Positive definite

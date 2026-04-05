"""
Reference tests for VBKalmanFilter.
Since filterpy doesn't have VB-AKF, we compare against a numpy implementation
based on the paper: https://users.aalto.fi/~ssarkka/pub/mvb-akf-mlsp.pdf

Algorithm (from Section III of the paper):
  Predict:
    m_{k|k-1} = F @ m_{k-1|k-1}
    P_{k|k-1} = F @ P_{k-1|k-1} @ F^T + Q

    nu_{k|k-1} = rho * (nu_{k-1|k-1} - d - 1) + d + 1
    V_{k|k-1}  = B @ V_{k-1|k-1} @ B^T

  Update (iterate N times):
    R_inv = (nu - d - 1) * inv(V)
    S = H @ P @ H^T + inv(R_inv)
    K = P @ H^T @ inv(S)
    m = m_{k|k-1} + K @ (y - H @ m_{k|k-1})
    P = P_{k|k-1} - K @ S @ K^T
    V = V_{k|k-1} + (y - H @ m) @ (y - H @ m)^T + H @ P @ H^T

  nu_{k|k} = nu_{k|k-1} + 1
"""
import torch
import numpy as np
import pytest
from kalman.vkf import VBKalmanFilter
from kalman.gaussian import GaussianState


def vbakf_reference_numpy(F, H, Q, R_init, rho, observations, n_iter=5):
    """
    Pure NumPy implementation of VB-AKF from the paper for comparison.
    """
    state_dim = F.shape[0]
    obs_dim = H.shape[0]
    B = np.sqrt(rho) * np.eye(obs_dim)

    # Initialize
    m = np.zeros(state_dim)
    P = np.eye(state_dim)
    nu = obs_dim + 2
    V = (nu - obs_dim - 1) * R_init

    T = observations.shape[0]
    means = np.zeros((T, state_dim))
    covs = np.zeros((T, state_dim, state_dim))

    for t in range(T):
        y = observations[t]

        # --- Predict ---
        m_pred = F @ m
        P_pred = F @ P @ F.T + Q

        nu_pred = rho * (nu - obs_dim - 1) + obs_dim + 1
        V_pred = B @ V @ B.T

        # --- Update (variational iterations) ---
        m_upd = m_pred.copy()
        P_upd = P_pred.copy()
        nu_upd = nu_pred + 1
        V_upd = V_pred.copy()

        for _ in range(n_iter):
            R_inv = (nu_upd - obs_dim - 1) * np.linalg.inv(V_upd)
            S = H @ P_pred @ H.T + np.linalg.inv(R_inv)
            K = P_pred @ H.T @ np.linalg.inv(S)

            m_upd = m_pred + K @ (y - H @ m_pred)
            P_upd = P_pred - K @ S @ K.T

            residual = y - H @ m_upd
            V_upd = V_pred + np.outer(residual, residual) + H @ P_upd @ H.T

        m = m_upd
        P = P_upd
        nu = nu_upd
        V = V_upd

        means[t] = m
        covs[t] = P

    return means, covs


class TestVBKFvsReference:

    def test_static_system(self):
        """Simple static system: F=I, H=[1,0], Q small."""
        state_dim, obs_dim = 2, 1
        F = np.eye(state_dim)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(state_dim) * 0.01
        R_init = np.eye(obs_dim) * 1.0
        rho = 0.95

        np.random.seed(42)
        T = 20
        true_state = np.array([3.0, 0.0])
        observations = true_state[0] + np.random.randn(T, obs_dim) * 1.0

        # Reference
        ref_means, ref_covs = vbakf_reference_numpy(F, H, Q, R_init, rho, observations)

        # Ours
        vkf = VBKalmanFilter(
            process_matrix=torch.tensor(F, dtype=torch.float64),
            measurement_matrix=torch.tensor(H, dtype=torch.float64),
            process_noise=torch.tensor(Q, dtype=torch.float64),
            initial_measurement_cov=torch.tensor(R_init, dtype=torch.float64),
            rho=rho,
        )
        means, covs = vkf(torch.tensor(observations, dtype=torch.float64))

        assert torch.allclose(
            means, torch.tensor(ref_means, dtype=torch.float64), atol=1e-6
        ), f"Means mismatch:\nours=\n{means}\nref=\n{ref_means}"
        assert torch.allclose(
            covs, torch.tensor(ref_covs, dtype=torch.float64), atol=1e-6
        ), f"Covs mismatch"

    def test_cv_model(self):
        """Constant-velocity model with VB adaptation."""
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R_init = np.eye(1) * 2.0
        rho = 0.9

        np.random.seed(7)
        T = 15
        observations = np.cumsum(np.random.randn(T, 1) * 0.5, axis=0)

        # Reference
        ref_means, ref_covs = vbakf_reference_numpy(F, H, Q, R_init, rho, observations)

        # Ours
        vkf = VBKalmanFilter(
            process_matrix=torch.tensor(F, dtype=torch.float64),
            measurement_matrix=torch.tensor(H, dtype=torch.float64),
            process_noise=torch.tensor(Q, dtype=torch.float64),
            initial_measurement_cov=torch.tensor(R_init, dtype=torch.float64),
            rho=rho,
        )
        means, covs = vkf(torch.tensor(observations, dtype=torch.float64))

        assert torch.allclose(
            means, torch.tensor(ref_means, dtype=torch.float64), atol=1e-6
        ), f"Means mismatch"
        assert torch.allclose(
            covs, torch.tensor(ref_covs, dtype=torch.float64), atol=1e-6
        ), f"Covs mismatch"

    def test_measurement_covariance_converges(self):
        """The VB estimate of R should converge near the true R."""
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.001
        true_R = 0.25
        R_init = np.eye(1) * 2.0  # Start far from true R
        rho = 0.99

        np.random.seed(123)
        T = 200
        observations = np.random.randn(T, 1) * np.sqrt(true_R)

        vkf = VBKalmanFilter(
            process_matrix=torch.tensor(F, dtype=torch.float64),
            measurement_matrix=torch.tensor(H, dtype=torch.float64),
            process_noise=torch.tensor(Q, dtype=torch.float64),
            initial_measurement_cov=torch.tensor(R_init, dtype=torch.float64),
            rho=rho,
        )
        vkf(torch.tensor(observations, dtype=torch.float64))

        estimated_R = vkf.get_measurement_covariance()
        # Should be within a factor of 3 of the true value
        assert estimated_R.item() > true_R * 0.3, f"R too small: {estimated_R.item()}"
        assert estimated_R.item() < true_R * 3.0, f"R too large: {estimated_R.item()}"

    def test_positive_definite_covariance(self):
        """Covariance should stay positive definite throughout."""
        F = np.eye(2)
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R_init = np.eye(1) * 0.5
        rho = 0.95

        np.random.seed(55)
        T = 50
        observations = np.random.randn(T, 1)

        vkf = VBKalmanFilter(
            process_matrix=torch.tensor(F, dtype=torch.float64),
            measurement_matrix=torch.tensor(H, dtype=torch.float64),
            process_noise=torch.tensor(Q, dtype=torch.float64),
            initial_measurement_cov=torch.tensor(R_init, dtype=torch.float64),
            rho=rho,
        )
        means, covs = vkf(torch.tensor(observations, dtype=torch.float64))

        assert not torch.isnan(means).any()
        assert not torch.isnan(covs).any()
        for t in range(T):
            eigvals = torch.linalg.eigvalsh(covs[t])
            assert torch.all(eigvals > 0), f"Non-PD covariance at step {t}: eigvals={eigvals}"

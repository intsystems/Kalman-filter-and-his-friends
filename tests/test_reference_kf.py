"""
Reference comparison tests for KalmanFilter against filterpy.
Uses a constant-velocity tracking model as test scenario.
"""
import torch
import numpy as np
import pytest
from filterpy.kalman import KalmanFilter as FilterpyKF
from kalman.filters import KalmanFilter
from kalman.gaussian import GaussianState


def make_cv_model():
    """Constant-velocity model: state = [pos, vel], obs = [pos]."""
    dt = 1.0
    F = np.array([[1, dt], [0, 1]], dtype=np.float64)
    H = np.array([[1, 0]], dtype=np.float64)
    Q = np.array([[0.25, 0.5], [0.5, 1.0]], dtype=np.float64) * 0.01
    R = np.array([[1.0]], dtype=np.float64)
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 10.0
    return F, H, Q, R, x0, P0


def make_filterpy_kf(F, H, Q, R, x0, P0):
    kf = FilterpyKF(dim_x=2, dim_z=1)
    kf.F = F.copy()
    kf.H = H.copy()
    kf.Q = Q.copy()
    kf.R = R.copy()
    kf.x = x0.reshape(-1, 1).copy()
    kf.P = P0.copy()
    return kf


def make_torch_kf(F, H, Q, R):
    return KalmanFilter(
        process_matrix=torch.tensor(F, dtype=torch.float64),
        measurement_matrix=torch.tensor(H, dtype=torch.float64),
        process_noise=torch.tensor(Q, dtype=torch.float64),
        measurement_noise=torch.tensor(R, dtype=torch.float64),
    )


class TestKFvsFilterpy:
    """Compare our KalmanFilter step-by-step against filterpy."""

    def test_single_predict(self):
        F, H, Q, R, x0, P0 = make_cv_model()

        # filterpy
        ref = make_filterpy_kf(F, H, Q, R, x0, P0)
        ref.predict()

        # ours
        kf = make_torch_kf(F, H, Q, R)
        state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )
        pred = kf.predict(state)

        assert torch.allclose(pred.mean, torch.tensor(ref.x.flatten(), dtype=torch.float64), atol=1e-10)
        assert torch.allclose(pred.covariance, torch.tensor(ref.P, dtype=torch.float64), atol=1e-10)

    def test_single_update(self):
        F, H, Q, R, x0, P0 = make_cv_model()
        z = np.array([[0.5]])

        # filterpy: predict then update
        ref = make_filterpy_kf(F, H, Q, R, x0, P0)
        ref.predict()
        ref_x_pred = ref.x.flatten().copy()
        ref_P_pred = ref.P.copy()
        ref.update(z)

        # ours
        kf = make_torch_kf(F, H, Q, R)
        state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )
        pred = kf.predict(state)
        upd = kf.update(pred, torch.tensor(z.flatten(), dtype=torch.float64))

        assert torch.allclose(upd.mean, torch.tensor(ref.x.flatten(), dtype=torch.float64), atol=1e-10)
        assert torch.allclose(upd.covariance, torch.tensor(ref.P, dtype=torch.float64), atol=1e-10)

    def test_multi_step_sequence(self):
        """Run 20 predict-update cycles and compare at every step."""
        F, H, Q, R, x0, P0 = make_cv_model()
        np.random.seed(42)
        measurements = np.random.randn(20, 1)

        # filterpy
        ref = make_filterpy_kf(F, H, Q, R, x0, P0)
        ref_means, ref_covs = [], []
        for z in measurements:
            ref.predict()
            ref.update(z.reshape(-1, 1))
            ref_means.append(ref.x.flatten().copy())
            ref_covs.append(ref.P.copy())

        # ours
        kf = make_torch_kf(F, H, Q, R)
        state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )
        for i, z in enumerate(measurements):
            state = kf.predict(state)
            state = kf.update(state, torch.tensor(z.flatten(), dtype=torch.float64))
            assert torch.allclose(
                state.mean,
                torch.tensor(ref_means[i], dtype=torch.float64),
                atol=1e-8,
            ), f"Mean mismatch at step {i}"
            assert torch.allclose(
                state.covariance,
                torch.tensor(ref_covs[i], dtype=torch.float64),
                atol=1e-8,
            ), f"Cov mismatch at step {i}"

    def test_2d_observation(self):
        """Test with state_dim=3, obs_dim=2."""
        dt = 1.0
        F = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        H = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float64)
        Q = np.eye(3) * 0.01
        R = np.eye(2) * 0.5
        x0 = np.zeros(3)
        P0 = np.eye(3) * 5.0

        np.random.seed(7)
        measurements = np.random.randn(15, 2)

        # filterpy
        ref = FilterpyKF(dim_x=3, dim_z=2)
        ref.F, ref.H, ref.Q, ref.R = F.copy(), H.copy(), Q.copy(), R.copy()
        ref.x = x0.reshape(-1, 1).copy()
        ref.P = P0.copy()

        # ours
        kf = KalmanFilter(
            process_matrix=torch.tensor(F, dtype=torch.float64),
            measurement_matrix=torch.tensor(H, dtype=torch.float64),
            process_noise=torch.tensor(Q, dtype=torch.float64),
            measurement_noise=torch.tensor(R, dtype=torch.float64),
        )
        state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )

        for z in measurements:
            ref.predict()
            ref.update(z.reshape(-1, 1))
            state = kf.predict(state)
            state = kf.update(state, torch.tensor(z, dtype=torch.float64))
            assert torch.allclose(
                state.mean,
                torch.tensor(ref.x.flatten(), dtype=torch.float64),
                atol=1e-8,
            )
            assert torch.allclose(
                state.covariance,
                torch.tensor(ref.P, dtype=torch.float64),
                atol=1e-8,
            )

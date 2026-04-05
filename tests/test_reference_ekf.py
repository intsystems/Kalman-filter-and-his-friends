"""
Reference comparison tests for ExtendedKalmanFilter against filterpy.
Uses nonlinear tracking scenarios.
"""
import torch
import numpy as np
import pytest
from filterpy.kalman import ExtendedKalmanFilter as FilterpyEKF
from kalman.extended import ExtendedKalmanFilter
from kalman.gaussian import GaussianState


class TestEKFvsFilterpy:
    """Compare our EKF against filterpy on a simple nonlinear system."""

    @staticmethod
    def _make_scenario():
        """
        Nonlinear system:
          f(x) = [x0 + dt*x1, x1]       (nearly linear transition)
          h(x) = [sqrt(x0^2 + x1^2)]     (range measurement)
        """
        dt = 0.1
        state_dim, obs_dim = 2, 1

        def f_np(x):
            return np.array([x[0] + dt * x[1], x[1]])

        def h_np(x):
            return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2)])

        def F_jac_np(x):
            return np.array([[1.0, dt], [0.0, 1.0]])

        def H_jac_np(x):
            r = np.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-12
            return np.array([[x[0] / r, x[1] / r]])

        def f_torch(x):
            return torch.stack([x[..., 0] + dt * x[..., 1], x[..., 1]], dim=-1)

        def h_torch(x):
            return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2).unsqueeze(-1)

        def F_jac_torch(x):
            batch = x.shape[:-1]
            J = torch.zeros(*batch, 2, 2, dtype=x.dtype)
            J[..., 0, 0] = 1.0
            J[..., 0, 1] = dt
            J[..., 1, 1] = 1.0
            return J

        def H_jac_torch(x):
            r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2) + 1e-12
            batch = x.shape[:-1]
            J = torch.zeros(*batch, 1, 2, dtype=x.dtype)
            J[..., 0, 0] = x[..., 0] / r
            J[..., 0, 1] = x[..., 1] / r
            return J

        Q = np.eye(2) * 0.01
        R = np.array([[0.5]])
        x0 = np.array([5.0, 1.0])
        P0 = np.eye(2) * 1.0

        return dict(
            state_dim=state_dim,
            obs_dim=obs_dim,
            f_np=f_np,
            h_np=h_np,
            F_jac_np=F_jac_np,
            H_jac_np=H_jac_np,
            f_torch=f_torch,
            h_torch=h_torch,
            F_jac_torch=F_jac_torch,
            H_jac_torch=H_jac_torch,
            Q=Q,
            R=R,
            x0=x0,
            P0=P0,
        )

    def test_single_predict_update(self):
        s = self._make_scenario()

        # --- filterpy ---
        ref = FilterpyEKF(dim_x=2, dim_z=1)
        ref.x = s["x0"].reshape(-1, 1).copy()
        ref.P = s["P0"].copy()
        ref.Q = s["Q"].copy()
        ref.R = s["R"].copy()
        ref.F = s["F_jac_np"](s["x0"])
        ref.x = s["f_np"](s["x0"]).reshape(-1, 1)
        ref.P = ref.F @ s["P0"] @ ref.F.T + s["Q"]

        z = np.array([[5.1]])
        ref.update(z, HJacobian=lambda x: s["H_jac_np"](x.flatten()),
                   Hx=lambda x: s["h_np"](x.flatten()).reshape(-1, 1))

        # --- ours ---
        ekf = ExtendedKalmanFilter(
            state_dim=s["state_dim"],
            obs_dim=s["obs_dim"],
            f=s["f_torch"],
            h=s["h_torch"],
            F_jacobian=s["F_jac_torch"],
            H_jacobian=s["H_jac_torch"],
            Q=torch.tensor(s["Q"], dtype=torch.float64),
            R=torch.tensor(s["R"], dtype=torch.float64),
            eps=0.0,  # disable jitter for exact comparison
        )
        state = GaussianState(
            torch.tensor(s["x0"], dtype=torch.float64),
            torch.tensor(s["P0"], dtype=torch.float64),
        )
        pred = ekf.predict(state)
        upd = ekf.update(pred, torch.tensor([5.1], dtype=torch.float64))

        assert torch.allclose(
            upd.mean,
            torch.tensor(ref.x.flatten(), dtype=torch.float64),
            atol=1e-8,
        ), f"Mean: ours={upd.mean}, ref={ref.x.flatten()}"
        assert torch.allclose(
            upd.covariance,
            torch.tensor(ref.P, dtype=torch.float64),
            atol=1e-8,
        ), f"Cov: ours=\n{upd.covariance}\nref=\n{ref.P}"

    def test_multi_step_sequence(self):
        """Run 15 steps and compare at each."""
        s = self._make_scenario()
        np.random.seed(99)
        measurements = 5.0 + np.random.randn(15, 1) * 0.5

        # --- filterpy ---
        ref = FilterpyEKF(dim_x=2, dim_z=1)
        ref.x = s["x0"].reshape(-1, 1).copy()
        ref.P = s["P0"].copy()
        ref.Q = s["Q"].copy()
        ref.R = s["R"].copy()

        ref_means, ref_covs = [], []
        for z in measurements:
            # predict
            ref.F = s["F_jac_np"](ref.x.flatten())
            ref.x = s["f_np"](ref.x.flatten()).reshape(-1, 1)
            ref.P = ref.F @ ref.P @ ref.F.T + ref.Q
            # update
            ref.update(
                z.reshape(-1, 1),
                HJacobian=lambda x: s["H_jac_np"](x.flatten()),
                Hx=lambda x: s["h_np"](x.flatten()).reshape(-1, 1),
            )
            ref_means.append(ref.x.flatten().copy())
            ref_covs.append(ref.P.copy())

        # --- ours ---
        ekf = ExtendedKalmanFilter(
            state_dim=s["state_dim"],
            obs_dim=s["obs_dim"],
            f=s["f_torch"],
            h=s["h_torch"],
            F_jacobian=s["F_jac_torch"],
            H_jacobian=s["H_jac_torch"],
            Q=torch.tensor(s["Q"], dtype=torch.float64),
            R=torch.tensor(s["R"], dtype=torch.float64),
            eps=0.0,
        )
        state = GaussianState(
            torch.tensor(s["x0"], dtype=torch.float64),
            torch.tensor(s["P0"], dtype=torch.float64),
        )
        for i, z in enumerate(measurements):
            state = ekf.predict(state)
            state = ekf.update(state, torch.tensor(z.flatten(), dtype=torch.float64))
            assert torch.allclose(
                state.mean,
                torch.tensor(ref_means[i], dtype=torch.float64),
                atol=1e-7,
            ), f"Mean mismatch at step {i}"
            assert torch.allclose(
                state.covariance,
                torch.tensor(ref_covs[i], dtype=torch.float64),
                atol=1e-7,
            ), f"Cov mismatch at step {i}"

    def test_ekf_on_linear_matches_kf(self):
        """On a linear system, EKF should match standard KF exactly."""
        from kalman.filters import KalmanFilter

        F_np = np.array([[1.0, 0.5], [0.0, 1.0]])
        H_np = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[0.5]])
        x0 = np.zeros(2)
        P0 = np.eye(2)

        def f_torch(x):
            F = torch.tensor(F_np, dtype=x.dtype)
            return (F @ x.unsqueeze(-1)).squeeze(-1)

        def h_torch(x):
            H = torch.tensor(H_np, dtype=x.dtype)
            return (H @ x.unsqueeze(-1)).squeeze(-1)

        def F_jac(x):
            return torch.tensor(F_np, dtype=x.dtype).expand(*x.shape[:-1], 2, 2)

        def H_jac(x):
            return torch.tensor(H_np, dtype=x.dtype).expand(*x.shape[:-1], 1, 2)

        ekf = ExtendedKalmanFilter(
            2, 1, f_torch, h_torch,
            F_jacobian=F_jac, H_jacobian=H_jac,
            Q=torch.tensor(Q, dtype=torch.float64),
            R=torch.tensor(R, dtype=torch.float64),
            eps=0.0,
        )
        kf = KalmanFilter(
            torch.tensor(F_np, dtype=torch.float64),
            torch.tensor(H_np, dtype=torch.float64),
            torch.tensor(Q, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
        )

        ekf_state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )
        kf_state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )

        np.random.seed(12)
        for _ in range(10):
            z = torch.tensor(np.random.randn(1), dtype=torch.float64)
            ekf_state = ekf.predict(ekf_state)
            kf_state = kf.predict(kf_state)
            ekf_state = ekf.update(ekf_state, z)
            kf_state = kf.update(kf_state, z)

            assert torch.allclose(ekf_state.mean, kf_state.mean, atol=1e-10)
            assert torch.allclose(ekf_state.covariance, kf_state.covariance, atol=1e-10)

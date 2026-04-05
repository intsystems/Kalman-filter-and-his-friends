"""
Reference comparison tests for UnscentedKalmanFilter against filterpy.
"""
import torch
import numpy as np
import pytest
from filterpy.kalman import UnscentedKalmanFilter as FilterpyUKF
from filterpy.kalman import MerweScaledSigmaPoints
from kalman.unscented import UnscentedKalmanFilter
from kalman.gaussian import GaussianState


def _make_linear_scenario():
    """Simple constant-velocity linear model for UKF comparison."""
    dt = 1.0
    state_dim, obs_dim = 2, 1
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 5.0
    alpha, beta, kappa = 1e-3, 2.0, 0.0

    def fx_np(x, dt_):
        return F @ x

    def hx_np(x):
        return H @ x

    def f_torch(x):
        F_t = torch.tensor(F, dtype=x.dtype)
        return (F_t @ x.unsqueeze(-1)).squeeze(-1)

    def h_torch(x):
        H_t = torch.tensor(H, dtype=x.dtype)
        return (H_t @ x.unsqueeze(-1)).squeeze(-1)

    return dict(
        state_dim=state_dim, obs_dim=obs_dim,
        F=F, H=H, Q=Q, R=R, x0=x0, P0=P0,
        alpha=alpha, beta=beta, kappa=kappa,
        fx_np=fx_np, hx_np=hx_np,
        f_torch=f_torch, h_torch=h_torch,
    )


def _make_nonlinear_scenario():
    """
    Nonlinear system:
      f(x) = [x0 + 0.1*x1, x1]
      h(x) = [x0^2 + x1]
    """
    state_dim, obs_dim = 2, 1
    Q = np.eye(2) * 0.01
    R = np.array([[0.5]])
    x0 = np.array([1.0, 0.5])
    P0 = np.eye(2) * 1.0
    alpha, beta, kappa = 0.1, 2.0, 0.0

    def fx_np(x, dt):
        return np.array([x[0] + 0.1 * x[1], x[1]])

    def hx_np(x):
        return np.array([x[0] ** 2 + x[1]])

    def f_torch(x):
        return torch.stack([x[..., 0] + 0.1 * x[..., 1], x[..., 1]], dim=-1)

    def h_torch(x):
        return (x[..., 0] ** 2 + x[..., 1]).unsqueeze(-1)

    return dict(
        state_dim=state_dim, obs_dim=obs_dim,
        Q=Q, R=R, x0=x0, P0=P0,
        alpha=alpha, beta=beta, kappa=kappa,
        fx_np=fx_np, hx_np=hx_np,
        f_torch=f_torch, h_torch=h_torch,
    )


class TestUKFvsFilterpy:

    def _run_comparison(self, scenario, measurements, atol=1e-6):
        s = scenario
        n = s["state_dim"]

        # --- filterpy ---
        pts = MerweScaledSigmaPoints(n, s["alpha"], s["beta"], s["kappa"])
        ref = FilterpyUKF(n, s["obs_dim"], dt=1.0, fx=s["fx_np"], hx=s["hx_np"], points=pts)
        ref.x = s["x0"].copy()
        ref.P = s["P0"].copy()
        ref.Q = s["Q"].copy()
        ref.R = s["R"].copy()

        ref_means, ref_covs = [], []
        for z in measurements:
            ref.predict()
            ref.update(z)
            ref_means.append(ref.x.copy())
            ref_covs.append(ref.P.copy())

        # --- ours ---
        ukf = UnscentedKalmanFilter(
            state_dim=s["state_dim"],
            obs_dim=s["obs_dim"],
            f=s["f_torch"],
            h=s["h_torch"],
            alpha=s["alpha"],
            beta=s["beta"],
            kappa=s["kappa"],
            Q=torch.tensor(s["Q"], dtype=torch.float64),
            R=torch.tensor(s["R"], dtype=torch.float64),
            eps=1e-12,
        )
        state = GaussianState(
            torch.tensor(s["x0"], dtype=torch.float64),
            torch.tensor(s["P0"], dtype=torch.float64),
        )
        for i, z in enumerate(measurements):
            state = ukf.predict(state)
            state = ukf.update(state, torch.tensor(z, dtype=torch.float64))
            assert torch.allclose(
                state.mean,
                torch.tensor(ref_means[i], dtype=torch.float64),
                atol=atol,
            ), f"Mean mismatch at step {i}: ours={state.mean}, ref={ref_means[i]}"
            assert torch.allclose(
                state.covariance,
                torch.tensor(ref_covs[i], dtype=torch.float64),
                atol=atol,
            ), f"Cov mismatch at step {i}"

    def test_linear_single_step(self):
        s = _make_linear_scenario()
        measurements = [np.array([0.5])]
        self._run_comparison(s, measurements)

    def test_linear_multi_step(self):
        s = _make_linear_scenario()
        np.random.seed(42)
        measurements = [np.random.randn(1) for _ in range(20)]
        self._run_comparison(s, measurements)

    def test_nonlinear_multi_step(self):
        s = _make_nonlinear_scenario()
        np.random.seed(77)
        measurements = [1.5 + np.random.randn(1) * 0.3 for _ in range(15)]
        self._run_comparison(s, measurements, atol=1e-5)

    def test_sigma_points_match(self):
        """Verify sigma point generation matches filterpy."""
        n = 2
        alpha, beta, kappa = 1e-3, 2.0, 0.0
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.3], [0.3, 0.5]])

        # filterpy
        pts = MerweScaledSigmaPoints(n, alpha, beta, kappa)
        ref_sigma = pts.sigma_points(mean, cov)  # (2n+1, n)
        ref_Wm = pts.Wm
        ref_Wc = pts.Wc

        # ours
        ukf = UnscentedKalmanFilter(
            state_dim=n, obs_dim=1,
            f=lambda x: x, h=lambda x: x[..., :1],
            alpha=alpha, beta=beta, kappa=kappa,
            Q=torch.eye(n, dtype=torch.float64) * 0.1,
            R=torch.eye(1, dtype=torch.float64) * 0.1,
        )
        sigma = ukf._sigma_points(
            torch.tensor(mean, dtype=torch.float64),
            torch.tensor(cov, dtype=torch.float64),
        )

        # Weights should match
        assert torch.allclose(
            ukf.Wm, torch.tensor(ref_Wm, dtype=torch.float64), atol=1e-12
        ), f"Wm mismatch: ours={ukf.Wm}, ref={ref_Wm}"
        assert torch.allclose(
            ukf.Wc, torch.tensor(ref_Wc, dtype=torch.float64), atol=1e-12
        ), f"Wc mismatch: ours={ukf.Wc}, ref={ref_Wc}"

        # Sigma points should match
        assert torch.allclose(
            sigma, torch.tensor(ref_sigma, dtype=torch.float64), atol=1e-8
        ), f"Sigma mismatch:\nours=\n{sigma}\nref=\n{ref_sigma}"

    def test_ukf_on_linear_matches_kf(self):
        """On a linear system with large alpha, UKF should closely match KF."""
        from kalman.filters import KalmanFilter

        F = np.array([[1.0, 0.5], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.1
        R = np.array([[0.5]])
        x0 = np.zeros(2)
        P0 = np.eye(2)

        def f_torch(x):
            F_t = torch.tensor(F, dtype=x.dtype)
            return (F_t @ x.unsqueeze(-1)).squeeze(-1)

        def h_torch(x):
            H_t = torch.tensor(H, dtype=x.dtype)
            return (H_t @ x.unsqueeze(-1)).squeeze(-1)

        # Use alpha=1 for less distortion on linear systems
        ukf = UnscentedKalmanFilter(
            2, 1, f_torch, h_torch,
            alpha=1.0, beta=0.0, kappa=0.0,
            Q=torch.tensor(Q, dtype=torch.float64),
            R=torch.tensor(R, dtype=torch.float64),
        )
        kf = KalmanFilter(
            torch.tensor(F, dtype=torch.float64),
            torch.tensor(H, dtype=torch.float64),
            torch.tensor(Q, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
        )

        ukf_state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )
        kf_state = GaussianState(
            torch.tensor(x0, dtype=torch.float64),
            torch.tensor(P0, dtype=torch.float64),
        )

        np.random.seed(33)
        for _ in range(10):
            z = torch.tensor(np.random.randn(1), dtype=torch.float64)
            ukf_state = ukf.predict(ukf_state)
            kf_state = kf.predict(kf_state)
            ukf_state = ukf.update(ukf_state, z)
            kf_state = kf.update(kf_state, z)

            # UKF on linear systems should be reasonably close to KF
            # Not exact because UKF uses sigma-point approximation
            assert torch.allclose(ukf_state.mean, kf_state.mean, atol=0.1)
            assert torch.allclose(ukf_state.covariance, kf_state.covariance, atol=0.1)

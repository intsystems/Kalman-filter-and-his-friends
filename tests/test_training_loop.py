"""
Training loop tests for all Kalman filters.

Verifies that each filter:
1. Has trainable parameters registered as nn.Parameter
2. Gradients flow through predict/update operations
3. An optimizer can update the parameters
4. Loss decreases when training on synthetic data
"""
import torch
import torch.nn as nn
import numpy as np
import pytest
from kalman.filters import KalmanFilter
from kalman.extended import ExtendedKalmanFilter
from kalman.unscented import UnscentedKalmanFilter
from kalman.vkf import VBKalmanFilter
from kalman.dkf import DeepKalmanFilter
from kalman.gaussian import GaussianState


# ================================================================== #
#                  Synthetic data generation                          #
# ================================================================== #
def generate_linear_data(T=50, state_dim=2, obs_dim=1, seed=42):
    """
    Generate data from a known linear system:
      x_{t+1} = F_true @ x_t + w_t,   w ~ N(0, Q_true)
      y_t     = H_true @ x_t + v_t,   v ~ N(0, R_true)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    F_true = torch.tensor([[1.0, 0.1], [0.0, 1.0]], dtype=torch.float64)
    H_true = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    Q_true = torch.eye(state_dim, dtype=torch.float64) * 0.01
    R_true = torch.eye(obs_dim, dtype=torch.float64) * 0.5

    states = [torch.zeros(state_dim, dtype=torch.float64)]
    observations = []
    for t in range(T):
        x = F_true @ states[-1] + torch.randn(state_dim, dtype=torch.float64) * 0.1
        y = H_true @ x + torch.randn(obs_dim, dtype=torch.float64) * np.sqrt(0.5)
        states.append(x)
        observations.append(y)

    return (
        torch.stack(observations),   # (T, obs_dim)
        torch.stack(states[1:]),     # (T, state_dim)
        F_true, H_true, Q_true, R_true,
    )


# ================================================================== #
#                  KalmanFilter training loop                         #
# ================================================================== #
class TestKFTrainingLoop:

    def test_has_parameters(self):
        kf = KalmanFilter(torch.eye(2), torch.eye(2), 0.01 * torch.eye(2), 0.1 * torch.eye(2))
        params = list(kf.named_parameters())
        names = {n for n, _ in params}
        assert "process_matrix" in names
        assert "measurement_matrix" in names
        assert "_process_noise._L" in names
        assert "_measurement_noise._L" in names

    def test_gradients_flow(self):
        """Gradients should reach F, H, Q, R through predict+update."""
        kf = KalmanFilter(
            torch.eye(2, dtype=torch.float64),
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            0.01 * torch.eye(2, dtype=torch.float64),
            0.1 * torch.eye(1, dtype=torch.float64),
        )

        state = GaussianState(
            torch.zeros(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )

        measurement = torch.tensor([1.0], dtype=torch.float64)
        predicted = kf.predict(state)
        updated = kf.update(predicted, measurement)

        # Loss: MSE on predicted measurement
        y_pred = (kf.measurement_matrix @ updated.mean.unsqueeze(-1)).squeeze(-1)
        loss = ((y_pred - measurement) ** 2).sum()
        loss.backward()

        for name, param in kf.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_training_loop_converges(self):
        """Train KF to learn the correct F matrix from observations."""
        observations, states, F_true, H_true, Q_true, R_true = generate_linear_data()

        # Initialize KF with wrong F (perturbed)
        F_init = F_true + torch.randn_like(F_true) * 0.3
        kf = KalmanFilter(F_init, H_true.clone(), Q_true.clone(), R_true.clone())

        # Only optimize F
        for p in kf.parameters():
            p.requires_grad_(False)
        kf.process_matrix.requires_grad_(True)

        optimizer = torch.optim.Adam([kf.process_matrix], lr=0.01)

        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            state = GaussianState(
                torch.zeros(2, dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
            )
            total_loss = torch.tensor(0.0, dtype=torch.float64)
            for t in range(len(observations)):
                state = kf.predict(state)
                state = kf.update(state, observations[t])
                # NLL-like loss: squared prediction error
                total_loss = total_loss + ((state.mean - states[t]) ** 2).sum()

            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Loss should decrease
        assert np.mean(losses[-10:]) < np.mean(losses[:10]), (
            f"Loss did not decrease: {np.mean(losses[:10]):.4f} -> {np.mean(losses[-10:]):.4f}"
        )


# ================================================================== #
#                  EKF training loop                                  #
# ================================================================== #
class TestEKFTrainingLoop:

    def test_has_parameters(self):
        ekf = ExtendedKalmanFilter(2, 1, lambda x: x, lambda x: x[..., :1])
        params = {n for n, _ in ekf.named_parameters()}
        assert "_Q_spd._L" in params
        assert "_R_spd._L" in params
        assert "_init_mean" in params
        assert "_init_cov_spd._L" in params

    def test_gradients_flow_through_predict_update(self):
        """Gradients should reach Q and R."""
        def f(x):
            return x

        def h(x):
            return x[..., :1]

        ekf = ExtendedKalmanFilter(
            2, 1, f, h,
            Q=torch.eye(2, dtype=torch.float64) * 0.1,
            R=torch.eye(1, dtype=torch.float64) * 0.5,
            eps=0.0,
        )

        state = GaussianState(
            torch.zeros(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )
        measurement = torch.tensor([1.0], dtype=torch.float64)

        predicted = ekf.predict(state)
        updated = ekf.update(predicted, measurement)

        loss = (updated.mean ** 2).sum() + (updated.covariance ** 2).sum()
        loss.backward()

        assert ekf._Q_spd._L.grad is not None
        assert ekf._R_spd._L.grad is not None

    def test_training_loop_learns_Q(self):
        """Train EKF to learn process noise Q from data."""
        torch.manual_seed(42)
        T = 30
        Q_true = torch.eye(2, dtype=torch.float64) * 0.05
        R_true = torch.eye(1, dtype=torch.float64) * 0.3

        F_mat = torch.tensor([[1.0, 0.1], [0.0, 1.0]], dtype=torch.float64)
        H_mat = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

        # Generate data
        states = [torch.zeros(2, dtype=torch.float64)]
        obs = []
        for _ in range(T):
            x = F_mat @ states[-1] + torch.randn(2, dtype=torch.float64) * 0.05**0.5
            y = H_mat @ x + torch.randn(1, dtype=torch.float64) * 0.3**0.5
            states.append(x)
            obs.append(y)
        obs = torch.stack(obs)
        states = torch.stack(states[1:])

        def f(x):
            return (F_mat @ x.unsqueeze(-1)).squeeze(-1)
        def h(x):
            return (H_mat @ x.unsqueeze(-1)).squeeze(-1)
        def F_jac(x):
            return F_mat.expand(*x.shape[:-1], 2, 2)
        def H_jac(x):
            return H_mat.expand(*x.shape[:-1], 1, 2)

        # Initialize with wrong Q
        ekf = ExtendedKalmanFilter(
            2, 1, f, h,
            F_jacobian=F_jac, H_jacobian=H_jac,
            Q=torch.eye(2, dtype=torch.float64) * 0.5,  # Wrong!
            R=R_true.clone(),
            eps=0.0,
        )

        for p in ekf.parameters():
            p.requires_grad_(False)
        ekf._Q_spd._L.requires_grad_(True)

        optimizer = torch.optim.Adam([ekf._Q_spd._L], lr=0.01)

        losses = []
        for epoch in range(80):
            optimizer.zero_grad()
            state = GaussianState(
                torch.zeros(2, dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
            )
            loss = torch.tensor(0.0, dtype=torch.float64)
            for t in range(T):
                state = ekf.predict(state)
                state = ekf.update(state, obs[t])
                loss = loss + ((state.mean - states[t]) ** 2).sum()

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert np.mean(losses[-10:]) < np.mean(losses[:10])


# ================================================================== #
#                  UKF training loop                                  #
# ================================================================== #
class TestUKFTrainingLoop:

    def test_has_parameters(self):
        ukf = UnscentedKalmanFilter(
            2, 1, lambda x: x, lambda x: x[..., :1],
            Q=torch.eye(2) * 0.1, R=torch.eye(1) * 0.1,
        )
        params = {n for n, _ in ukf.named_parameters()}
        assert "_Q_spd._L" in params
        assert "_R_spd._L" in params
        assert "_init_mean" in params
        assert "_init_cov_spd._L" in params

    def test_gradients_flow(self):
        """Gradients should reach Q and R through sigma-point operations."""
        ukf = UnscentedKalmanFilter(
            2, 1,
            f=lambda x: x * 1.1,
            h=lambda x: x[..., :1],
            Q=torch.eye(2, dtype=torch.float64) * 0.1,
            R=torch.eye(1, dtype=torch.float64) * 0.5,
        )

        state = GaussianState(
            torch.ones(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )
        predicted = ukf.predict(state)
        updated = ukf.update(predicted, torch.tensor([1.0], dtype=torch.float64))

        loss = (updated.mean ** 2).sum() + (updated.covariance ** 2).sum()
        loss.backward()

        assert ukf._Q_spd._L.grad is not None
        assert ukf._R_spd._L.grad is not None

    def test_training_loop_converges(self):
        """Train UKF to learn R from data."""
        torch.manual_seed(42)
        T = 30

        F_mat = torch.tensor([[1.0, 0.1], [0.0, 1.0]], dtype=torch.float64)
        H_mat = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        Q_true = torch.eye(2, dtype=torch.float64) * 0.01
        R_true = torch.eye(1, dtype=torch.float64) * 0.3

        states = [torch.zeros(2, dtype=torch.float64)]
        obs = []
        for _ in range(T):
            x = F_mat @ states[-1] + torch.randn(2, dtype=torch.float64) * 0.1
            y = H_mat @ x + torch.randn(1, dtype=torch.float64) * 0.3**0.5
            states.append(x)
            obs.append(y)
        obs = torch.stack(obs)
        states = torch.stack(states[1:])

        def f(x):
            return (F_mat @ x.unsqueeze(-1)).squeeze(-1)
        def h(x):
            return (H_mat @ x.unsqueeze(-1)).squeeze(-1)

        ukf = UnscentedKalmanFilter(
            2, 1, f, h,
            alpha=1.0, beta=0.0, kappa=0.0,
            Q=Q_true.clone(),
            R=torch.eye(1, dtype=torch.float64) * 2.0,  # Wrong R
        )

        for p in ukf.parameters():
            p.requires_grad_(False)
        ukf._R_spd._L.requires_grad_(True)

        optimizer = torch.optim.Adam([ukf._R_spd._L], lr=0.01)

        losses = []
        for epoch in range(80):
            optimizer.zero_grad()
            state = GaussianState(
                torch.zeros(2, dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
            )
            loss = torch.tensor(0.0, dtype=torch.float64)
            for t in range(T):
                state = ukf.predict(state)
                state = ukf.update(state, obs[t])
                loss = loss + ((state.mean - states[t]) ** 2).sum()

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert np.mean(losses[-10:]) < np.mean(losses[:10])


# ================================================================== #
#                  VBKalmanFilter training loop                       #
# ================================================================== #
class TestVBKFTrainingLoop:

    def test_has_parameters(self):
        vkf = VBKalmanFilter(
            torch.eye(2), torch.tensor([[1.0, 0.0]]),
            0.1 * torch.eye(2), torch.eye(1),
        )
        params = {n for n, _ in vkf.named_parameters()}
        assert "process_matrix" in params
        assert "measurement_matrix" in params
        assert "_process_noise_spd._L" in params

    def test_gradients_flow(self):
        """Gradients should reach process_matrix through predict+update."""
        vkf = VBKalmanFilter(
            torch.eye(2, dtype=torch.float64),
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            0.1 * torch.eye(2, dtype=torch.float64),
            torch.eye(1, dtype=torch.float64),
        )

        state = GaussianState(
            torch.zeros(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
        )

        predicted = vkf.predict(state)
        updated = vkf.update(predicted, torch.tensor([1.0], dtype=torch.float64))

        loss = (updated.mean ** 2).sum()
        loss.backward()

        assert vkf.process_matrix.grad is not None
        assert vkf.measurement_matrix.grad is not None
        # process_noise gradient flows through SPDParameter
        assert vkf._process_noise_spd._L.grad is not None

    def test_training_loop_converges(self):
        """Train VBF process_matrix from data."""
        torch.manual_seed(42)
        T = 30

        F_true = torch.eye(2, dtype=torch.float64)
        H_true = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

        states = [torch.zeros(2, dtype=torch.float64)]
        obs = []
        for _ in range(T):
            x = F_true @ states[-1] + torch.randn(2, dtype=torch.float64) * 0.1
            y = H_true @ x + torch.randn(1, dtype=torch.float64) * 0.5
            states.append(x)
            obs.append(y)
        obs = torch.stack(obs)
        states = torch.stack(states[1:])

        vkf = VBKalmanFilter(
            F_true + torch.randn(2, 2, dtype=torch.float64) * 0.3,
            H_true.clone(),
            0.1 * torch.eye(2, dtype=torch.float64),
            torch.eye(1, dtype=torch.float64),
        )

        for p in vkf.parameters():
            p.requires_grad_(False)
        vkf.process_matrix.requires_grad_(True)

        optimizer = torch.optim.Adam([vkf.process_matrix], lr=0.01)

        losses = []
        for epoch in range(80):
            optimizer.zero_grad()

            # Reset VBF internal state for each epoch
            vkf.nu = vkf.obs_dim + 2
            vkf.V = (vkf.nu - vkf.obs_dim - 1) * torch.eye(vkf.obs_dim, dtype=torch.float64)

            state = GaussianState(
                torch.zeros(2, dtype=torch.float64),
                torch.eye(2, dtype=torch.float64),
            )
            loss = torch.tensor(0.0, dtype=torch.float64)
            for t in range(T):
                state = vkf.predict(state)
                state = vkf.update(state, obs[t])
                loss = loss + ((state.mean - states[t]) ** 2).sum()

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert np.mean(losses[-10:]) < np.mean(losses[:10])


# ================================================================== #
#                  DKF training loop (already tested, quick check)    #
# ================================================================== #
class TestDKFTrainingLoop:

    def test_has_parameters(self):
        params_cfg = {
            "dataset": "test", "transition_type": "mlp", "transition_layers": 1,
            "emission_type": "mlp", "emission_layers": 1, "data_type": "real",
            "dim_hidden": 16, "var_model": "LR", "rnn_size": 16, "rnn_layers": 1,
            "inference_model": "mean_field", "use_prev_input": False,
        }
        dkf = DeepKalmanFilter(4, 8, params_cfg)
        n_params = sum(p.numel() for p in dkf.parameters())
        assert n_params > 0

    def test_full_training_loop(self):
        """End-to-end: create model, build optimizer, train, check loss decreases."""
        torch.manual_seed(0)
        params_cfg = {
            "dataset": "test", "transition_type": "mlp", "transition_layers": 1,
            "emission_type": "mlp", "emission_layers": 1, "data_type": "real",
            "dim_hidden": 32, "var_model": "L", "rnn_size": 32, "rnn_layers": 1,
            "inference_model": "mean_field", "use_prev_input": False,
        }
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=params_cfg)
        model.anneal = 1.0
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        T, B = 15, 4
        obs = torch.randn(T, B, 8)
        mask = torch.ones(T, B)

        losses = []
        for step in range(30):
            optimizer.zero_grad()
            loss, nll, kl = model.loss(obs, mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert np.mean(losses[-5:]) < np.mean(losses[:5])

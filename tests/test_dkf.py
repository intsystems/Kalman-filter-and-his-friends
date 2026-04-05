"""
Tests for DeepKalmanFilter.

DKF is a VAE-based model with no direct filterpy equivalent.
We test:
1. Construction and forward pass for all configurations
2. ELBO loss correctness (manual KL + NLL vs. model output)
3. Gradient flow (all parameters receive gradients)
4. Training convergence on synthetic data
5. Inference network produces valid posteriors
"""
import torch
import numpy as np
import pytest
from kalman.dkf import DeepKalmanFilter
from kalman.gaussian import GaussianState


def _make_params(
    transition_type="mlp",
    var_model="LR",
    data_type="real",
    inference_model="mean_field",
    dim_hidden=16,
    rnn_size=16,
):
    return {
        "dataset": "test",
        "transition_type": transition_type,
        "transition_layers": 1,
        "emission_type": "mlp",
        "emission_layers": 1,
        "data_type": data_type,
        "dim_hidden": dim_hidden,
        "var_model": var_model,
        "rnn_size": rnn_size,
        "rnn_layers": 1,
        "inference_model": inference_model,
        "use_prev_input": False,
    }


class TestDKFConstruction:
    """Test that DKF builds correctly for all config variants."""

    @pytest.mark.parametrize("transition_type", ["mlp", "simple_gated"])
    @pytest.mark.parametrize("var_model", ["L", "R", "LR"])
    def test_build_variants(self, transition_type, var_model):
        params = _make_params(transition_type=transition_type, var_model=var_model)
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=params)
        assert model.state_dim == 4
        assert model.obs_dim == 8
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_build_binary(self):
        params = _make_params(data_type="binary")
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=params)
        assert hasattr(model, "p_emis_out")

    def test_build_structured_inference(self):
        params = _make_params(inference_model="structured")
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=params)
        assert hasattr(model, "q_st")


class TestDKFForward:
    """Test forward pass shapes and validity."""

    @pytest.fixture
    def model(self):
        return DeepKalmanFilter(state_dim=4, obs_dim=8, params=_make_params())

    def test_forward_shapes(self, model):
        obs = torch.randn(10, 3, 8)  # (T, B, obs_dim)
        means, variances = model(obs)
        assert means.shape == (10, 3, 4)
        assert variances.shape == (10, 3, 4)

    def test_forward_finite(self, model):
        obs = torch.randn(5, 2, 8)
        means, variances = model(obs)
        assert torch.isfinite(means).all()
        assert torch.isfinite(variances).all()
        assert (variances > 0).all()  # variance must be positive

    def test_predict(self, model):
        state = GaussianState(torch.randn(3, 4), torch.ones(3, 4))
        pred = model.predict(state)
        assert pred.mean.shape == (3, 4)
        assert pred.covariance.shape == (3, 4)
        assert (pred.covariance > 0).all()

    def test_update(self, model):
        state = GaussianState(torch.randn(3, 4), torch.ones(3, 4))
        upd = model.update(state, torch.randn(3, 8))
        assert upd.mean.shape == (3, 4)
        assert upd.covariance.shape == (3, 4)
        assert (upd.covariance > 0).all()

    def test_emission(self, model):
        state = GaussianState(torch.randn(3, 4), torch.ones(3, 4))
        mu, var = model._get_emission(state)
        assert mu.shape == (3, 8)
        assert var.shape == (3, 8)
        assert (var > 0).all()

    def test_emission_binary(self):
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=_make_params(data_type="binary"))
        state = GaussianState(torch.randn(3, 4), torch.ones(3, 4))
        prob, _ = model._get_emission(state)
        assert prob.shape == (3, 8)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_infer(self, model):
        obs = torch.randn(10, 3, 8)
        z, mu, log_var = model.infer(obs)
        assert z.shape == (10, 3, 4)
        assert mu.shape == (10, 3, 4)
        assert log_var.shape == (10, 3, 4)
        assert torch.isfinite(z).all()
        assert torch.isfinite(mu).all()
        assert torch.isfinite(log_var).all()


class TestDKFLoss:
    """Test ELBO loss computation correctness."""

    @pytest.fixture
    def model(self):
        return DeepKalmanFilter(state_dim=4, obs_dim=8, params=_make_params())

    def test_loss_runs(self, model):
        obs = torch.randn(10, 3, 8)
        mask = torch.ones(10, 3)
        loss, nll, kl = model.loss(obs, mask)
        assert torch.isfinite(loss)
        assert torch.isfinite(nll)
        assert torch.isfinite(kl)
        assert kl >= 0  # KL divergence is non-negative

    def test_loss_with_mask(self, model):
        """Loss should be lower when fewer timesteps are valid."""
        obs = torch.randn(10, 3, 8)
        mask_full = torch.ones(10, 3)
        mask_half = torch.ones(10, 3)
        mask_half[5:] = 0

        loss_full, _, _ = model.loss(obs, mask_full)
        loss_half, _, _ = model.loss(obs, mask_half)
        # Half-masked loss should be smaller since fewer terms contribute
        assert loss_half < loss_full

    def test_loss_binary(self):
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=_make_params(data_type="binary"))
        obs = torch.rand(10, 3, 8)  # binary observations in [0, 1]
        mask = torch.ones(10, 3)
        loss, nll, kl = model.loss(obs, mask)
        assert torch.isfinite(loss)
        assert nll >= 0

    def test_kl_manual_verification(self, model):
        """Verify KL divergence against manual computation."""
        torch.manual_seed(42)
        obs = torch.randn(5, 2, 8)
        mask = torch.ones(5, 2)

        q_mu, q_var = model._run_inference_network(obs)
        eps = torch.randn_like(q_mu)
        z = q_mu + torch.sqrt(q_var) * eps

        # For t=0: p = N(0, I)
        p_mu = torch.zeros_like(z)
        p_var = torch.ones_like(z)

        # KL(q || p) for diagonal Gaussians
        kl_manual = 0.5 * (
            torch.log(p_var) - torch.log(q_var) - 1.0
            + q_var / p_var
            + (p_mu - q_mu) ** 2 / p_var
        ).sum(-1)  # (T, B)

        kl_total = (kl_manual * mask).sum()

        # Run model loss with same eps (synthetic dataset -> no transition)
        model_synthetic = DeepKalmanFilter(
            state_dim=4, obs_dim=8, params=_make_params()
        )
        # Copy inference weights
        model_synthetic.load_state_dict(model.state_dict())
        model_synthetic.params["dataset"] = "synthetic_test"

        _, _, kl_from_loss = model_synthetic.loss(obs, mask, eps=eps)
        assert torch.allclose(kl_from_loss, kl_total, atol=1e-4)


class TestDKFGradients:
    """Test that gradients flow to all parameters."""

    def test_gradients_flow(self):
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=_make_params())
        obs = torch.randn(5, 2, 8)
        mask = torch.ones(5, 2)

        loss, _, _ = model.loss(obs, mask)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    @pytest.mark.parametrize("transition_type", ["mlp", "simple_gated"])
    def test_gradients_all_transitions(self, transition_type):
        params = _make_params(transition_type=transition_type)
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=params)
        obs = torch.randn(5, 2, 8)
        mask = torch.ones(5, 2)

        loss, _, _ = model.loss(obs, mask)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestDKFTraining:
    """Test that DKF training actually decreases loss."""

    def test_training_decreases_loss(self):
        """Train for a few steps on synthetic data and verify loss decreases."""
        torch.manual_seed(0)
        state_dim, obs_dim = 4, 8
        params = _make_params(dim_hidden=32, rnn_size=32)
        model = DeepKalmanFilter(state_dim=state_dim, obs_dim=obs_dim, params=params)
        model.anneal = 1.0  # full KL from start

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate simple synthetic data: sine waves
        T, B = 20, 8
        t = torch.linspace(0, 4 * np.pi, T).unsqueeze(-1).unsqueeze(-1)
        freqs = torch.randn(1, B, obs_dim) * 0.5
        obs = torch.sin(t * freqs) + torch.randn(T, B, obs_dim) * 0.1
        mask = torch.ones(T, B)

        # Collect losses
        losses = []
        for step in range(50):
            optimizer.zero_grad()
            loss, nll, kl = model.loss(obs, mask)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at end should be lower than at start (average of first 5 vs last 5)
        early_loss = np.mean(losses[:5])
        late_loss = np.mean(losses[-5:])
        assert late_loss < early_loss, (
            f"Loss did not decrease: {early_loss:.2f} -> {late_loss:.2f}"
        )

    def test_train_step_method(self):
        """Test the train_step convenience method."""
        model = DeepKalmanFilter(state_dim=4, obs_dim=8, params=_make_params())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        obs = torch.randn(10, 3, 8)
        mask = torch.ones(10, 3)

        optimizer.zero_grad()
        loss, nll, kl = model.train_step(obs, mask)
        optimizer.step()

        assert isinstance(loss, float)
        assert isinstance(nll, float)
        assert isinstance(kl, float)
        assert np.isfinite(loss)

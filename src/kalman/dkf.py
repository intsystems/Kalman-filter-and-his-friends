from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kalman.gaussian import GaussianState
from kalman.filters import BaseFilter


class DeepKalmanFilter(BaseFilter):
    """
    Deep Kalman Filter (Krishnan et al., 2015).

    Combines a generative model (transition + emission neural networks)
    with an inference (recognition) network for approximate posterior estimation
    of latent states in a sequential VAE framework.

    Note: This filter uses *diagonal* Gaussians for the latent space.
    GaussianState.covariance stores the diagonal variance vector (not a matrix).

    Parameters
    ----------
    state_dim : int
        Dimension of the latent state z.
    obs_dim : int
        Dimension of observations x.
    params : dict
        Configuration dictionary with keys:
        - dataset : str
        - transition_type : 'mlp' | 'simple_gated'
        - transition_layers : int
        - emission_type : 'mlp'
        - emission_layers : int
        - data_type : 'real' | 'binary'
        - dim_hidden : int
        - var_model : 'L' | 'R' | 'LR'
        - rnn_size : int
        - rnn_layers : int
        - inference_model : 'mean_field' | 'structured'
        - use_prev_input : bool
    smooth : bool
        Placeholder for future smoother support.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        params: Dict[str, Any],
        smooth: bool = False,
    ):
        super().__init__(state_dim, obs_dim, smooth)
        self.params = params

        self._create_generative_params()
        self._create_inference_params()

        # KL annealing
        self.anneal = 0.01
        self.update_ctr = 1

    # ------------------------------------------------------------------ #
    #                     parameter construction                          #
    # ------------------------------------------------------------------ #
    def _create_generative_params(self) -> None:
        """Create generative model parameters (transition + emission)."""
        if "synthetic" in self.params["dataset"]:
            return

        # --- Transition ---
        if self.params["transition_type"] == "mlp":
            self.p_trans = nn.ModuleList()
            for idx in range(self.params["transition_layers"]):
                in_dim = self.state_dim if idx == 0 else self.params["dim_hidden"] * 2
                self.p_trans.append(nn.Linear(in_dim, self.params["dim_hidden"] * 2))
            self.p_trans_W_mu = nn.Linear(self.params["dim_hidden"] * 2, self.state_dim)
            self.p_trans_W_cov = nn.Linear(self.params["dim_hidden"] * 2, self.state_dim)

        elif self.params["transition_type"] == "simple_gated":
            self.p_gate_embed = nn.Sequential(
                nn.Linear(self.state_dim, self.params["dim_hidden"] * 2),
                nn.Tanh(),
                nn.Linear(self.params["dim_hidden"] * 2, self.state_dim),
            )
            self.p_z_proj = nn.Sequential(
                nn.Linear(self.state_dim, self.params["dim_hidden"] * 2),
                nn.Tanh(),
                nn.Linear(self.params["dim_hidden"] * 2, self.state_dim),
            )
            self.p_trans_W_mu = nn.Linear(self.state_dim, self.state_dim)
            with torch.no_grad():
                self.p_trans_W_mu.weight.copy_(torch.eye(self.state_dim))
                self.p_trans_W_mu.bias.zero_()
            self.p_trans_W_cov = nn.Linear(self.state_dim, self.state_dim)

        # --- Emission ---
        if self.params["emission_type"] == "mlp":
            self.p_emis = nn.ModuleList()
            for idx in range(self.params["emission_layers"]):
                in_dim = self.state_dim if idx == 0 else self.params["dim_hidden"]
                self.p_emis.append(nn.Linear(in_dim, self.params["dim_hidden"]))
            if self.params["data_type"] == "binary":
                self.p_emis_out = nn.Linear(self.params["dim_hidden"], self.obs_dim)
            elif self.params["data_type"] == "real":
                self.p_emis_mu = nn.Linear(self.params["dim_hidden"], self.obs_dim)
                self.p_emis_cov = nn.Linear(self.params["dim_hidden"], self.obs_dim)

    def _create_inference_params(self) -> None:
        """Create inference (recognition) network parameters."""
        self.q_input = nn.Linear(self.obs_dim, self.params["rnn_size"])

        if self.params["var_model"] in ("L", "LR"):
            self.lstm_l = nn.LSTM(
                input_size=self.params["rnn_size"],
                hidden_size=self.params["rnn_size"],
                num_layers=self.params["rnn_layers"],
                batch_first=False,
            )
        if self.params["var_model"] in ("R", "LR"):
            self.lstm_r = nn.LSTM(
                input_size=self.params["rnn_size"],
                hidden_size=self.params["rnn_size"],
                num_layers=self.params["rnn_layers"],
                batch_first=False,
            )

        if self.params["inference_model"] == "structured":
            self.q_st = nn.Sequential(
                nn.Linear(self.state_dim, self.params["rnn_size"]),
                nn.Tanh(),
            )

        self.q_mu = nn.Linear(self.params["rnn_size"], self.state_dim)
        self.q_cov = nn.Linear(self.params["rnn_size"], self.state_dim)

        if self.params["var_model"] == "LR" and self.params["inference_model"] == "mean_field":
            self.q_mu_r = nn.Linear(self.params["rnn_size"], self.state_dim)
            self.q_cov_r = nn.Linear(self.params["rnn_size"], self.state_dim)

    # ------------------------------------------------------------------ #
    #                     generative model helpers                        #
    # ------------------------------------------------------------------ #
    def _get_transition(
        self,
        state: GaussianState,
        measurement: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """
        Compute transition distribution p(z_t | z_{t-1}).

        Returns GaussianState with diagonal variance (stored in .covariance as a vector).
        """
        z = state.mean

        if self.params["transition_type"] == "mlp":
            h = z
            for layer in self.p_trans:
                h = torch.tanh(layer(h))
            mu = self.p_trans_W_mu(h)
            var = F.softplus(self.p_trans_W_cov(h))

        elif self.params["transition_type"] == "simple_gated":
            gate = torch.sigmoid(self.p_gate_embed(z))
            z_prop = self.p_z_proj(z)
            mu = gate * z_prop + (1 - gate) * self.p_trans_W_mu(z)
            var = F.softplus(self.p_trans_W_cov(torch.tanh(z_prop)))

        return GaussianState(mu, var)

    def _get_emission(
        self, state: GaussianState
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute emission distribution p(x_t | z_t).

        Returns (mu, var) for real data or (prob, None) for binary data.
        """
        z = state.mean
        h = z
        for layer in self.p_emis:
            h = torch.tanh(layer(h))

        if self.params["data_type"] == "binary":
            return torch.sigmoid(self.p_emis_out(h)), None
        elif self.params["data_type"] == "real":
            mu = self.p_emis_mu(h)
            var = F.softplus(self.p_emis_cov(h))
            return mu, var

    # ------------------------------------------------------------------ #
    #                    inference network                                #
    # ------------------------------------------------------------------ #
    def _run_inference_network(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the recognition network on a full observation sequence.

        Parameters
        ----------
        observations : torch.Tensor
            Shape (T, B, obs_dim).

        Returns
        -------
        q_mu : torch.Tensor, shape (T, B, state_dim)
        q_var : torch.Tensor, shape (T, B, state_dim)  (diagonal variance)
        """
        # Input embedding: (T, B, rnn_size)
        h = torch.tanh(self.q_input(observations))

        if self.params["var_model"] == "LR":
            h_l, _ = self.lstm_l(h)
            h_r, _ = self.lstm_r(torch.flip(h, [0]))
            h_r = torch.flip(h_r, [0])
            if self.params["inference_model"] == "mean_field":
                q_mu = self.q_mu(h_l) + self.q_mu_r(h_r)
                q_var = F.softplus(self.q_cov(h_l) + self.q_cov_r(h_r))
            else:
                h_comb = h_l + h_r
                q_mu = self.q_mu(h_comb)
                q_var = F.softplus(self.q_cov(h_comb))
        elif self.params["var_model"] == "L":
            h_l, _ = self.lstm_l(h)
            q_mu = self.q_mu(h_l)
            q_var = F.softplus(self.q_cov(h_l))
        elif self.params["var_model"] == "R":
            h_r, _ = self.lstm_r(torch.flip(h, [0]))
            h_r = torch.flip(h_r, [0])
            q_mu = self.q_mu(h_r)
            q_var = F.softplus(self.q_cov(h_r))

        return q_mu, q_var

    # ------------------------------------------------------------------ #
    #                     predict / update / forward                      #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        state: GaussianState,
        measurement: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Predict next latent state using the transition network."""
        return self._get_transition(state, measurement)

    def update(
        self,
        state: GaussianState,
        measurement: torch.Tensor,
    ) -> GaussianState:
        """
        Update step: run inference network on a single observation.

        Parameters
        ----------
        measurement : torch.Tensor, shape (B, obs_dim)
        """
        # Add time dim for LSTM: (1, B, obs_dim)
        if measurement.dim() == 2:
            measurement = measurement.unsqueeze(0)

        h = torch.tanh(self.q_input(measurement))

        if self.params["var_model"] == "LR":
            h_l, _ = self.lstm_l(h)
            h_r, _ = self.lstm_r(h)
            h_out = h_l + h_r
        elif self.params["var_model"] == "L":
            h_out, _ = self.lstm_l(h)
        elif self.params["var_model"] == "R":
            h_out, _ = self.lstm_r(h)

        h_out = h_out.squeeze(0)  # (B, rnn_size)
        mu = self.q_mu(h_out)
        var = F.softplus(self.q_cov(h_out))
        return GaussianState(mu, var)

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: run inference network on the entire sequence.

        Parameters
        ----------
        observations : torch.Tensor
            Shape (T, B, obs_dim).

        Returns
        -------
        means : torch.Tensor, shape (T, B, state_dim)
        variances : torch.Tensor, shape (T, B, state_dim)
        """
        q_mu, q_var = self._run_inference_network(observations)
        return q_mu, q_var

    # ------------------------------------------------------------------ #
    #                           loss                                      #
    # ------------------------------------------------------------------ #
    def loss(
        self,
        observations: torch.Tensor,
        mask: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the negative ELBO = NLL + anneal * KL.

        Parameters
        ----------
        observations : torch.Tensor
            Shape (T, B, obs_dim).
        mask : torch.Tensor
            Shape (T, B). Binary mask for valid timesteps.
        eps : torch.Tensor, optional
            Reparameterization noise, shape (T, B, state_dim).
            If None, sampled automatically.

        Returns
        -------
        loss, nll, kl : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        T, B = observations.shape[:2]

        # 1. Run inference network to get q(z_t | x_{1:T})
        q_mu, q_var = self._run_inference_network(observations)  # (T, B, state_dim)

        # 2. Sample z from q using reparameterization
        if eps is None:
            eps = torch.randn_like(q_mu)
        z = q_mu + torch.sqrt(q_var) * eps  # (T, B, state_dim)

        # 3. Compute transition prior p(z_t | z_{t-1})
        #    For t=0, use standard normal prior N(0, I)
        p_mu = torch.zeros_like(z)
        p_var = torch.ones_like(z)

        if T > 1 and "synthetic" not in self.params["dataset"]:
            z_prev = z[:-1]  # (T-1, B, state_dim)
            state_prev = GaussianState(z_prev, torch.ones_like(z_prev))
            trans = self._get_transition(state_prev)
            p_mu[1:] = trans.mean
            p_var[1:] = trans.covariance

        # 4. Compute emission p(x_t | z_t)
        z_state = GaussianState(z, torch.ones_like(z))
        obs_mu, obs_var = self._get_emission(z_state)

        # 5. Negative log-likelihood
        if self.params["data_type"] == "binary":
            nll_per_step = F.binary_cross_entropy(
                obs_mu, observations, reduction="none"
            ).sum(-1)  # (T, B)
        else:
            nll_per_step = 0.5 * (
                np.log(2 * np.pi)
                + torch.log(obs_var)
                + (observations - obs_mu) ** 2 / obs_var
            ).sum(-1)  # (T, B)

        nll = (nll_per_step * mask).sum()

        # 6. KL divergence: KL(q(z) || p(z))  (diagonal Gaussians)
        kl_per_step = 0.5 * (
            torch.log(p_var) - torch.log(q_var) - 1.0
            + q_var / p_var
            + (p_mu - q_mu) ** 2 / p_var
        ).sum(-1)  # (T, B)

        kl = (kl_per_step * mask).sum()

        loss = nll + self.anneal * kl
        return loss, nll, kl

    def train_step(
        self,
        observations: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Single training step (forward + backward, no optimizer step).

        Parameters
        ----------
        observations : torch.Tensor, shape (T, B, obs_dim)
        mask : torch.Tensor, shape (T, B)

        Returns
        -------
        loss, nll, kl as float values.
        """
        loss, nll, kl = self.loss(observations, mask)
        loss.backward()

        self.update_ctr += 1
        if self.update_ctr % 1000 == 0:
            self.anneal = min(1.0, 0.01 + self.update_ctr / 1000)

        return loss.item(), nll.item(), kl.item()

    def infer(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Posterior inference: sample latent states from q(z | x).

        Parameters
        ----------
        observations : torch.Tensor, shape (T, B, obs_dim)

        Returns
        -------
        z : torch.Tensor, shape (T, B, state_dim) -- sampled latent states
        mu : torch.Tensor, shape (T, B, state_dim) -- posterior means
        log_var : torch.Tensor, shape (T, B, state_dim) -- log posterior variances
        """
        with torch.no_grad():
            q_mu, q_var = self._run_inference_network(observations)
            eps = torch.randn_like(q_mu)
            z = q_mu + torch.sqrt(q_var) * eps
            return z, q_mu, torch.log(q_var)

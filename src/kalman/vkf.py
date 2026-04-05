# https://users.aalto.fi/~ssarkka/pub/mvb-akf-mlsp.pdf

from typing import Optional, Tuple
import torch
from torch import nn
from kalman.gaussian import GaussianState
from kalman.filters import BaseFilter, SPDParameter


class VBKalmanFilter(BaseFilter):
    """
    Variational Bayesian Adaptive Kalman Filter (VB-AKF).

    Adaptively estimates the measurement noise covariance R
    using a variational Bayesian approach with an inverse-Wishart prior.
    """

    def __init__(
        self,
        process_matrix: torch.Tensor,
        measurement_matrix: torch.Tensor,
        process_noise: torch.Tensor,
        initial_measurement_cov: torch.Tensor,
        rho: float = 0.95,
        B: Optional[torch.Tensor] = None,
        state_dim: Optional[int] = None,
        obs_dim: Optional[int] = None,
        n_iter: int = 5,
    ):
        if state_dim is None:
            state_dim = process_matrix.shape[-1]
        if obs_dim is None:
            obs_dim = measurement_matrix.shape[-2]

        super().__init__(state_dim, obs_dim)

        self.process_matrix = nn.Parameter(process_matrix.clone())       # F
        self.measurement_matrix = nn.Parameter(measurement_matrix.clone())  # H
        self._process_noise_spd = SPDParameter(process_noise.clone())    # Q
        self.rho = rho
        self.n_iter = n_iter

        if B is not None:
            self.B = B
        else:
            self.B = torch.sqrt(torch.tensor(rho, dtype=process_matrix.dtype)) * torch.eye(
                obs_dim, dtype=process_matrix.dtype
            )

        # Inverse-Wishart parameters
        self.nu = obs_dim + 2  # degrees of freedom
        self.V = (self.nu - obs_dim - 1) * initial_measurement_cov  # scale matrix

    @property
    def process_noise(self) -> torch.Tensor:
        return self._process_noise_spd()

    def predict(
        self,
        state: GaussianState,
        process_matrix: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        """Prediction step with Wishart parameter dynamics."""
        F = process_matrix if process_matrix is not None else self.process_matrix
        Q = self.process_noise

        predicted_mean = (F @ state.mean.unsqueeze(-1)).squeeze(-1)
        predicted_cov = F @ state.covariance @ F.mT + Q

        self.nu = self.rho * (self.nu - self.obs_dim - 1) + self.obs_dim + 1
        self.V = self.B @ self.V @ self.B.mT

        return GaussianState(predicted_mean, predicted_cov)

    def update(
        self,
        state: GaussianState,
        measurement: torch.Tensor,
    ) -> GaussianState:
        """Iterative variational update step."""
        H = self.measurement_matrix
        y = measurement
        m_pred = state.mean
        P_pred = state.covariance

        nu = self.nu + 1
        V = self.V.clone()
        m = m_pred.clone()
        P = P_pred.clone()

        for _ in range(self.n_iter):
            R_inv = (nu - self.obs_dim - 1) * torch.linalg.inv(V)
            S = H @ P_pred @ H.mT + torch.linalg.inv(R_inv)
            K = P_pred @ H.mT @ torch.linalg.inv(S)

            innovation = y - (H @ m_pred.unsqueeze(-1)).squeeze(-1)
            m = m_pred + (K @ innovation.unsqueeze(-1)).squeeze(-1)
            P = P_pred - K @ S @ K.mT

            residual = y - (H @ m.unsqueeze(-1)).squeeze(-1)
            V = self.V + (residual.unsqueeze(-1) @ residual.unsqueeze(-2)) + H @ P @ H.mT

        self.nu = nu
        self.V = V
        return GaussianState(m, P)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = observations.shape[0]
        dtype, device = observations.dtype, observations.device
        means = torch.zeros(T, self.state_dim, dtype=dtype, device=device)
        covs = torch.zeros(T, self.state_dim, self.state_dim, dtype=dtype, device=device)

        current_state = GaussianState(
            torch.zeros(self.state_dim, dtype=dtype, device=device),
            torch.eye(self.state_dim, dtype=dtype, device=device),
        )
        for t in range(T):
            predicted_state = self.predict(current_state)
            updated_state = self.update(predicted_state, observations[t])
            means[t] = updated_state.mean
            covs[t] = updated_state.covariance
            current_state = updated_state
        return means, covs

    def get_measurement_covariance(self) -> torch.Tensor:
        return self.V / (self.nu - self.obs_dim - 1)

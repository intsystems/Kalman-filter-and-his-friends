from typing import Tuple, Optional
from overrides import override

import torch
from torch import nn
from typing import Optional

from kalman.gaussian import GaussianState


class BaseFilter(nn.Module):
    """
    Abstract base class for Kalman Filters
    """

    def __init__(self, state_dim: int, obs_dim: int, smooth: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.smooth = smooth

    def predict_(
        self, 
        state_mean: torch.Tensor, 
        state_cov: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal function for single-step predict.
        Returns:
            predicted_state_mean, predicted_state_cov
        """
        pass

    def update_(
        self, 
        state_mean: torch.Tensor, 
        state_cov: torch.Tensor, 
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal single-step update.
        Returns:
            updated_state_mean, updated_state_cov
        """
        pass
    
    def predict(self, state: GaussianState) -> GaussianState:
        """
        Single-step predict.
        Returns:
            GaussianStateю
        """
        m, P = self.predict_(state.mean, state.covariance)
        return GaussianState(m, P)
    
    def update(self, state: GaussianState, measurement: torch.Tensor) -> GaussianState:
        """
        Single-step update.
        Returns:
            GaussianStateю
        """
        m, P = self.update_(state.mean, state.covariance, measurement)
        return GaussianState(m, P)

    def predict_update(
        self, 
        state: GaussianState, 
        measurement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step predict and update in one function.
        Returns: 
            updated_state_mean, updated_state_cov
        """
        predicted_state = self.update(state)
        updated_state = self.predict(predicted_state, measurement)
        return updated_state

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes an entire sequence of observations in shape (T, B, obs_dim).
        Returns
        -------
        all_states, all_states : GaussianState ‑‑ convenient wrapper holding the whole trajectory
        """
        pass
        

class KalmanFilter(BaseFilter):
    """
    Kalman Filter class.
    Attributes:
        process_matrix (torch.Tensor): State transition matrix (F)
            Shape: (*, state_dim, state_dim)
        measurement_matrix (torch.Tensor): Projection matrix (H)
            Shape: (*, obs_dim, state_dim)
        process_noise (torch.Tensor): Uncertainty on the process (Q)
            Shape: (*, state_dim, state_dim)
        measurement_noise (torch.Tensor): Uncertainty on the measure (R)
            Shape: (*, obs_dim, obs_dim)
    """
    def __init__(self, 
                 process_matrix: torch.Tensor,
                 measurement_matrix: torch.Tensor,
                 process_noise: torch.Tensor,
                 measurement_noise: torch.Tensor):
        super().__init__(process_matrix.shape[-1], measurement_matrix.shape[-1])
        self.process_matrix = process_matrix
        self.measurement_matrix = measurement_matrix
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
    def predict(self,
        state: GaussianState,
        *,
        process_matrix: Optional[torch.Tensor] = None,
        process_noise: Optional[torch.Tensor] = None,) -> GaussianState:
        """
        Predict step of the Kalman Filter.
        Initially state_mean is (B, state_dim)
        """
        if process_matrix is None:
            process_matrix = self.process_matrix
        if process_noise is None:
            process_noise = self.process_noise

        state_mean = process_matrix @ state.mean.unsqueeze(-1)  # now it is (B, state_dim, 1)
        state_mean = state_mean.squeeze(-1)

        state_cov = (process_matrix @ state.covariance @ process_matrix.transpose(-2, -1)) + process_noise

        return GaussianState(state_mean, state_cov)

    def project(
        self,
        state: GaussianState,
        *,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,
        precompute_precision=True,
    ) -> GaussianState:
        """Project the current state (usually the prior) onto the measurement space
        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, bs, state_dim)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, obs_dim, obs_dim)
            precompute_precision (bool): Precompute precision matrix (inverse covariance)
                Done once to prevent more computations
                Default: True

        Returns:
            GaussianState: Prior on the next state

        """
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise

        mean = measurement_matrix @ state.mean.unsqueeze(-1)  # now it is (B, obs_dim, 1)
        mean = mean.squeeze(-1)

        cov = (measurement_matrix @ state.covariance @ measurement_matrix.transpose(-2, -1)) + measurement_noise

        precision = (
            torch.linalg.inv(cov).transpose(-2, -1)
            if precompute_precision
            else None
        )

        return GaussianState(mean, cov, precision)

    @override
    def update(self,
        state: GaussianState,
        measurement: torch.Tensor,
        *,
        projection: Optional[GaussianState] = None,
        measurement_matrix: Optional[torch.Tensor] = None,
        measurement_noise: Optional[torch.Tensor] = None,) -> GaussianState:
        """
        Update step of the Kalman Filter.
        """
        
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise

        if projection is None:
            projection = self.project(state, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise)

        residual = measurement - projection.mean  # now it is (B, obs_dim)

        kalman_gain = (
            state.covariance @ measurement_matrix.transpose(-2, -1) @ projection.precision
        )  # now it is (B, state_dim, obs_dim)

        updated_mean = state.mean + (kalman_gain @ residual.unsqueeze(-1)).squeeze(-1)  # now it is (B, state_dim)

        updated_cov = (
            state.covariance
            - kalman_gain @ measurement_matrix @ state.covariance
        )  # now it is (B, state_dim, state_dim)

        return GaussianState(updated_mean, updated_cov)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the Kalman Filter over a sequence of observations.
        """
        means = []
        covs = []
        state = self.predict(self.initial_state)
        for obs in observations:
            state = self.update(state, obs)
            means.append(state.mean)
            covs.append(state.covariance)   
        return GaussianState(torch.stack(means), torch.stack(covs))

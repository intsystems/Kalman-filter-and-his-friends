from typing import Tuple

import torch
from torch import nn
import dataclasses
import contextlib
from typing import Optional, overload

import torch
import torch.linalg

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

    def predict(
        self, 
        state_mean: torch.Tensor, 
        state_cov: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step predict.
        Returns:
            predicted_state_mean, predicted_state_cov
        """
        pass

    def update(
        self, 
        state_mean: torch.Tensor, 
        state_cov: torch.Tensor, 
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step update.
        Returns:
            updated_state_mean, updated_state_cov
        """
        pass
    
    def predict_update(
        self, 
        state_mean: torch.Tensor,
        state_cov: torch.Tensor,
        measurement: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step predict and update in one function.
        Returns: 
            updated_state_mean, updated_state_cov
        """
        updated_state_mean, updated_state_cov = self.update(state_mean, state_cov, measurement)
        predicted_state_mean, predicted_state_cov = self.predict(updated_state_mean, updated_state_cov)
        return predicted_state_mean, predicted_state_cov

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes an entire sequence of observations in shape (T, B, obs_dim).
        Returns all_states, pairs (all_means, all_covs) with shapes:
            all_means: (T, B, state_dim)
            all_covs:  (T, B, state_dim, state_dim)
        """
        pass


class KalmanFilter(BaseFilter):
    """
    Kalman Filter class.
    Attributes:
        process_matrix (torch.Tensor): State transition matrix (F)
            Shape: (*, dim_x, dim_x)
        measurement_matrix (torch.Tensor): Projection matrix (H)
            Shape: (*, dim_z, dim_x)
        process_noise (torch.Tensor): Uncertainty on the process (Q)
            Shape: (*, dim_x, dim_x)
        measurement_noise (torch.Tensor): Uncertainty on the measure (R)
            Shape: (*, dim_z, dim_z)
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
        """
        if process_matrix is None:
            process_matrix = self.process_matrix
        if process_noise is None:
            process_noise = self.process_noise
        state_mean = process_matrix @ state.mean
        state_cov = process_matrix @ state.covariance @ process_matrix.transpose(-2, -1) + process_noise
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

        Use the measurement equation: z_k = H x_k + N(0, R).
        Support batch computation: You can provide multiple measurements, projections models (H, R)
        or/and multiple states. You just need to ensure that shapes are broadcastable.

        Args:
            state (GaussianState): Current state estimation (Usually the results of `predict`)
            measurement_matrix (Optional[torch.Tensor]): Overwrite the default projection matrix
                Shape: (*, dim_z, dim_x)
            measurement_noise (Optional[torch.Tensor]): Overwrite the default projection noise)
                Shape: (*, dim_z, dim_z)
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

        mean = measurement_matrix @ state.mean
        covariance = measurement_matrix @ state.covariance @ measurement_matrix.mT + measurement_noise

        return GaussianState(
            mean,
            covariance,
            (
                covariance.inverse().mT
                if precompute_precision
                else None
            ),
        )
    

    def update(self,
        state: GaussianState,
        measure: torch.Tensor,
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
        if measurement_matrix is None:
            measurement_matrix = self.measurement_matrix
        if measurement_noise is None:
            measurement_noise = self.measurement_noise
        if projection is None:
            projection = self.project(state, measurement_matrix=measurement_matrix, measurement_noise=measurement_noise)

        residual = measure - projection.mean

        kalman_gain = state.covariance @ measurement_matrix.mT @ projection.precision

        mean = state.mean + kalman_gain @ residual

        covariance = state.covariance - kalman_gain @ measurement_matrix @ state.covariance

        return GaussianState(mean, covariance)

class ExtendedKalmanFilter(BaseFilter):
    """
    Extended Kalman Filter class.
    """
    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__(state_dim, obs_dim)


class UnscentedKalmanFilter(BaseFilter):
    """
    Unscented Kalman Filter class.
    """
    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__(state_dim, obs_dim)


class VariationalKalmanFilter(BaseFilter):
    """
    Variational Kalman Filter class.
    """
    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__(state_dim, obs_dim)
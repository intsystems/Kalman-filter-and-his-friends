from typing import Tuple

import torch
import torch import nn

class BaseFilter(nn.Module):
    """
    Abstract base class for Kalman Filters
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

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
        # ...
        return predicted_state_mean, predicted_state_cov

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes an entire sequence of observations in shape (T, B, obs_dim).
        Returns all_states, pairs (all_means, all_covs) with shapes:
            all_means: (T, B, state_dim)
            all_covs:  (T, B, state_dim, state_dim)
        """
        pass

# Author: torch-kf
# https://github.com/raphaelreme/torch-kf/blob/main/torch_kf/kalman_filter.py

import torch
from torch import nn
from typing import Optional, overload


class GaussianState:
    """
    Class for Gaussian state.

    Attributes:
        mean (torch.Tensor): Mean of the distribution
            Shape: (*, dim, 1)
        covariance (torch.Tensor): Covariance of the distribution
            Shape: (*, dim, dim)
        precision (Optional[torch.Tensor]): Optional inverse covariance matrix
            This may be useful for some computations (E.G mahalanobis distance, likelihood) after a predict step.
            Shape: (*, dim, dim)
    """
    def __init__(self, mean: torch.Tensor, covariance: torch.Tensor, precision: Optional[torch.Tensor] = None):
        self.mean = mean
        self.covariance = covariance
        self.precision = precision
    
    def clone(self) -> "GaussianState":
        """Clone the Gaussian State using `torch.Tensor.clone`

        Returns:
            GaussianState: A copy of the Gaussian state
        """
        return GaussianState(
            self.mean.clone(), self.covariance.clone(), self.precision.clone() if self.precision is not None else None
        )

    def __getitem__(self, idx) -> "GaussianState":
        return GaussianState(
            self.mean[idx], self.covariance[idx], self.precision[idx] if self.precision is not None else None
        )

    def __setitem__(self, idx, value) -> None:
        if isinstance(value, GaussianState):
            self.mean[idx] = value.mean
            self.covariance[idx] = value.covariance
            if self.precision is not None and value.precision is not None:
                self.precision[idx] = value.precision

            return

        raise NotImplementedError()

    @overload
    def to(self, dtype: torch.dtype) -> "GaussianState": ...

    @overload
    def to(self, device: torch.device) -> "GaussianState": ...

    def to(self, fmt):
        """Convert a GaussianState to a specific device or dtype

        Args:
            fmt (torch.dtype | torch.device): Memory format to send the state to.

        Returns:
            GaussianState: The GaussianState with the right format
        """
        return GaussianState(
            self.mean.to(fmt),
            self.covariance.to(fmt),
            self.precision.to(fmt) if self.precision is not None else None,
        )

    def mahalanobis_squared(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the squared mahalanobis distance of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Squared mahalanobis distance for each measure/state
                Shape: (*)
        """
        diff = self.mean - measure  # You are responsible for broadcast
        if self.precision is None:
            # The inverse is transposed (back) to be contiguous: as it is symmetric
            # This is equivalent and faster to hold on the contiguous verison
            # But this may slightly increase floating errors.
            self.precision = self.covariance.inverse().mT

        return (diff.mT @ self.precision @ diff)[..., 0, 0]  # Delete trailing dimensions

    def mahalanobis(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the mahalanobis distance of given measure

        Computations of the sqrt can be slow. If you want to compare with a given threshold,
        you should rather compare the squared mahalanobis with the squared threshold.

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Mahalanobis distance for each measure/state
                Shape: (*)
        """
        return self.mahalanobis_squared(measure).sqrt()

    def log_likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the log-likelihood of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Log-likelihood for each measure/state
                Shape: (*, 1)
        """
        maha_2 = self.mahalanobis_squared(measure)
        log_det = torch.log(torch.det(self.covariance))

        return -0.5 * (self.covariance.shape[-1] * torch.log(2 * torch.tensor(torch.pi)) + log_det + maha_2)

    def likelihood(self, measure: torch.Tensor) -> torch.Tensor:
        """Computes the likelihood of given measure

        It supports batch computation: You can provide multiple measurements and have multiple states
        You just need to ensure that shapes are broadcastable.

        Args:
            measure (torch.Tensor): Points to consider
                Shape: (*, dim, 1)

        Returns:
            torch.Tensor: Likelihood for each measure/state
                Shape: (*, 1)
        """
        return self.log_likelihood(measure).exp()
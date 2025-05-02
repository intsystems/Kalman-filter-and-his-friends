from typing import Tuple

import torch
import torch.linalg

from kalman.filters import BaseFilter
from kalman.gaussian import GaussianState

class UnscentedKalmanFilter(BaseFilter):
    """
    Unscented Kalman Filter class.
    Implements the UKF using the scaled unscented transformation.
    
    Attributes:
        alpha (float): Determines the spread of sigma points (usually 1e-3)
        beta (float): Incorporates prior knowledge of the distribution (2 for Gaussian)
        kappa (float): Secondary scaling parameter (usually 0)
    """
    def __init__(self, 
                 state_dim: int, 
                 obs_dim: int,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        super().__init__(state_dim, obs_dim)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Compute lambda parameter
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim
        
        # Weights for mean and covariance
        self.Wm = torch.zeros(2 * state_dim + 1)  # Mean weights
        self.Wc = torch.zeros(2 * state_dim + 1)  # Covariance weights
        
        # Initialize weights
        self._compute_weights()
    
    def _compute_weights(self):
        """Compute weights for sigma points."""
        state_dim = self.state_dim
        lambda_ = self.lambda_
        
        # Weight for the mean of the 0th sigma point
        self.Wm[0] = lambda_ / (state_dim + lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
        # Weights for other sigma points
        for i in range(1, 2 * state_dim + 1):
            self.Wm[i] = 1 / (2 * (state_dim + lambda_))
            self.Wc[i] = self.Wm[i]
    
    def _compute_sigma_points(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma points using the scaled unscented transformation.
        
        Args:
            mean: State mean (..., state_dim, 1)
            cov: State covariance (..., state_dim, state_dim)
            
        Returns:
            Sigma points (..., 2*state_dim+1, state_dim, 1)
        """
        state_dim = self.state_dim
        lambda_ = self.lambda_
        
        # Compute matrix square root of (state_dim + lambda) * cov
        # scaling = torch.linalg.cholesky((state_dim + lambda_) * cov)
        scaling = (state_dim + lambda_) * cov
        try:
            scaling = torch.linalg.cholesky(scaling)
        except RuntimeError:
            # If Cholesky fails, add small diagonal noise and try again
            eps = 1e-5 * torch.eye(state_dim, device=scaling.device, dtype=scaling.dtype)
            scaling = torch.linalg.cholesky(scaling + eps)
        
        # Create sigma points matrix with correct shape
        sigma_points = torch.zeros(*mean.shape[:-2], 2 * state_dim + 1, state_dim, 1, 
                                device=mean.device, dtype=mean.dtype)
        
        # First sigma point is the mean
        sigma_points[..., 0, :, :] = mean
        
        # Remaining sigma points
        for i in range(state_dim):
            # Get the i-th column of the scaling matrix and reshape properly
            col = scaling[..., :, i:i+1]  # shape: (..., state_dim, 1)
            
            # Add and subtract columns to create sigma points
            sigma_points[..., i+1, :, :] = mean + col
            sigma_points[..., state_dim+i+1, :, :] = mean - col
            
        return sigma_points
    
    def predict(
        self, 
        state: GaussianState,
        process_model: callable,
        process_noise: torch.Tensor
    ) -> GaussianState:
        """
        Predict step of the UKF.
        
        Args:
            state: Current state estimate
            process_model: Nonlinear state transition function
            process_noise: Process noise covariance
            
        Returns:
            Predicted state
        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(state.mean, state.covariance)
        
        # Propagate sigma points through process model
        predicted_sigma_points = process_model(sigma_points)
        
        # Compute predicted mean and covariance
        pred_mean = (self.Wm.reshape(-1, *([1]*(len(predicted_sigma_points.shape)-1))) * predicted_sigma_points).sum(dim=-3)
        
        # Compute predicted covariance
        diff = predicted_sigma_points - pred_mean.unsqueeze(-3)
        pred_cov = (self.Wc.reshape(-1, *([1]*(len(diff.shape)-1))) * (diff @ diff.transpose(-2, -1))).sum(dim=-3)
        pred_cov = pred_cov + process_noise
        
        return GaussianState(pred_mean, pred_cov)
    
    def update(
        self,
        state: GaussianState,
        measurement: torch.Tensor,
        measurement_model: callable,
        measurement_noise: torch.Tensor
    ) -> GaussianState:
        """
        Update step of the UKF.
        
        Args:
            state: Predicted state from predict step
            measurement: Current measurement
            measurement_model: Nonlinear measurement function
            measurement_noise: Measurement noise covariance
            
        Returns:
            Updated state estimate
        """
        # Generate sigma points
        sigma_points = self._compute_sigma_points(state.mean, state.covariance)
        
        # Propagate sigma points through measurement model
        measurement_sigma_points = measurement_model(sigma_points)
        
        # Compute predicted measurement mean and covariance
        meas_mean = (self.Wm.reshape(-1, *([1]*(len(measurement_sigma_points.shape)-1))) * measurement_sigma_points).sum(dim=-3)
        
        # Compute measurement covariance and cross-covariance
        meas_diff = measurement_sigma_points - meas_mean.unsqueeze(-3)
        state_diff = sigma_points - state.mean.unsqueeze(-3)
        
        meas_cov = (self.Wc.reshape(-1, *([1]*(len(meas_diff.shape)-1))) * 
                   (meas_diff @ meas_diff.transpose(-2, -1))).sum(dim=-3)
        meas_cov = meas_cov + measurement_noise
        
        cross_cov = (self.Wc.reshape(-1, *([1]*(len(state_diff.shape)-1))) * 
                    (state_diff @ meas_diff.transpose(-2, -1))).sum(dim=-3)
        
        # Compute Kalman gain
        kalman_gain = cross_cov @ torch.linalg.inv(meas_cov)
        
        # Compute updated state
        residual = measurement - meas_mean
        updated_mean = state.mean + kalman_gain @ residual
        updated_cov = state.covariance - kalman_gain @ meas_cov @ kalman_gain.transpose(-2, -1)
        
        return GaussianState(updated_mean, updated_cov)
    
    def predict_update(
        self,
        state: GaussianState,
        measurement: torch.Tensor,
        process_model: callable,
        process_noise: torch.Tensor,
        measurement_model: callable,
        measurement_noise: torch.Tensor
    ) -> GaussianState:
        """
        Combined predict and update steps.
        
        Args:
            state: Current state estimate
            measurement: Current measurement
            process_model: Nonlinear state transition function
            process_noise: Process noise covariance
            measurement_model: Nonlinear measurement function
            measurement_noise: Measurement noise covariance
            
        Returns:
            Updated state estimate
        """
        # Predict step
        predicted_state = self.predict(state, process_model, process_noise)
        
        # Update step
        updated_state = self.update(predicted_state, measurement, measurement_model, measurement_noise)
        
        return updated_state
    
    def forward(self, 
               observations: torch.Tensor,
               process_model: callable,
               process_noise: torch.Tensor,
               measurement_model: callable,
               measurement_noise: torch.Tensor,
               initial_state: GaussianState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes an entire sequence of observations.
        
        Args:
            observations: Measurement sequence (T, B, obs_dim)
            process_model: Nonlinear state transition function
            process_noise: Process noise covariance
            measurement_model: Nonlinear measurement function
            measurement_noise: Measurement noise covariance
            initial_state: Initial state estimate
            
        Returns:
            Tuple of (all_means, all_covs) with shapes:
                all_means: (T, B, state_dim, 1)
                all_covs: (T, B, state_dim, state_dim)
        """
        T = observations.shape[0]
        B = observations.shape[1] if len(observations.shape) > 1 else 1
        
        all_means = torch.zeros(T, B, self.state_dim, 1)
        all_covs = torch.zeros(T, B, self.state_dim, self.state_dim)
        
        current_state = initial_state
        
        for t in range(T):
            # Predict
            predicted_state = self.predict(current_state, process_model, process_noise)
            
            # Update
            measurement = observations[t]
            updated_state = self.update(predicted_state, measurement, measurement_model, measurement_noise)
            
            # Store results
            all_means[t] = updated_state.mean
            all_covs[t] = updated_state.covariance
            
            # Update current state
            current_state = updated_state
            
        return all_means, all_covs
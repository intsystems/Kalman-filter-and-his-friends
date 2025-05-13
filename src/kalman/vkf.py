# https://users.aalto.fi/~ssarkka/pub/mvb-akf-mlsp.pdf

from typing import Optional, Tuple
import torch
from torch import nn
from kalman.gaussian import GaussianState
from kalman.filters import BaseFilter

class VBKalmanFilter(BaseFilter):
    """
    Variational Bayesian Adaptive Kalman Filter (VB-AKF)
    """
    def __init__(self,
                 process_matrix: torch.Tensor,
                 measurement_matrix: torch.Tensor,
                 process_noise: torch.Tensor,
                 initial_measurement_cov: torch.Tensor,
                 rho: float = 0.95,
                 B: torch.Tensor = None,
                 state_dim: int = None,
                 obs_dim: int = None):
        
        if state_dim is None:
            state_dim = process_matrix.shape[-1]
        if obs_dim is None:
            obs_dim = measurement_matrix.shape[-2]
            
        super().__init__(state_dim, obs_dim)
        
        # Основные параметры
        self.process_matrix = process_matrix       # F (state_dim, state_dim)
        self.measurement_matrix = measurement_matrix # H (obs_dim, state_dim)
        self.process_noise = process_noise         # Q (state_dim, state_dim)
        
        # Параметры адаптации ковариации
        self.rho = rho
        self.B = B if B is not None else torch.sqrt(torch.tensor(rho)) * torch.eye(obs_dim)
        
        # Инициализация параметров обратного распределения Уишарта
        self.nu = obs_dim + 2  # Степени свободы
        self.V = (self.nu - obs_dim - 1) * initial_measurement_cov  # Масштабная матрица
        
    def predict(self,
               state: GaussianState,
               process_matrix: Optional[torch.Tensor] = None) -> GaussianState:
        """
        Prediction step with covariance dynamics
        """
        F = process_matrix if process_matrix is not None else self.process_matrix
        Q = self.process_noise
        # Predict state
        predicted_mean = state.mean @ F
        predicted_cov = F @ state.covariance @ F.T + Q

        # Predict covariance parameters
        self.nu = self.rho * (self.nu - self.obs_dim - 1) + self.obs_dim + 1
        self.V = self.B @ self.V @ self.B.T
        
        return GaussianState(predicted_mean, predicted_cov)
    
    def update(self,
              state: GaussianState,
              measurement: torch.Tensor) -> GaussianState:
        """
        Iterative variational update step
        """
        H = self.measurement_matrix
        y = measurement
        
        # Инициализация параметров
        m = state.mean.clone()
        P = state.covariance.clone()
        nu = self.nu + 1
        V = self.V.clone()
        for i in range(5):
            R_inv = (nu - self.obs_dim - 1) * torch.inverse(V)
            S = H @ P @ H.T + torch.inverse(R_inv)
            K = P @ H.T @ torch.inverse(S)
            
            m = state.mean + torch.einsum('hv,v->h', K, y - state.mean @ H.T)
            P = state.covariance - K @ S @ K.transpose(-1, -2)
            
            # Обновление параметров ковариации
            V = self.V + H @ P @ H.T + torch.einsum('i,j->ij', y - m @ H.T, y - m @ H.T)
        
        # Сохраняем новые параметры
        self.V = V
        
        return GaussianState(m, P)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process full sequence (T, B, obs_dim)
        """
        T, _ = observations.shape
        means = torch.zeros(T, self.state_dim)
        covs = torch.zeros(T, self.state_dim, self.state_dim)
        
        # Инициализация
        current_state = GaussianState(
            torch.zeros(self.state_dim),
            torch.eye(self.state_dim).repeat(1, 1))
        
        for t in range(T):
            # Predict
            predicted_state = self.predict(current_state)
            
            # Update
            updated_state = self.update(predicted_state, observations[t])
            
            # Store results
            means[t] = updated_state.mean
            covs[t] = updated_state.covariance
            current_state = updated_state
            
        return means, covs

    def get_measurement_covariance(self) -> torch.Tensor:
        """
        Returns current estimate of measurement covariance
        """
        return self.V / (self.nu - self.obs_dim - 1)

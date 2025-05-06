from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from kalman.gaussian import GaussianState
from kalman.filters import BaseFilter

class DeepKalmanFilter(BaseFilter):
    """
    Deep Kalman Filter implementation that combines neural networks with Kalman filtering.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 obs_dim: int,
                 params: Dict[str, Any],
                 smooth: bool = False):
        super().__init__(state_dim, obs_dim, smooth)
        self.params = params
        
        # Initialize parameters
        self._create_generative_params()
        self._create_inference_params()
        
        # Annealing
        self.anneal = 0.01
        self.update_ctr = 1
    
    def _get_weight(self, shape: Tuple[int, ...], scale: float = 0.01) -> nn.Parameter:
        """Initialize weights"""
        return nn.Parameter(torch.randn(shape) * scale)
    
    def _create_generative_params(self) -> None:
        """Create generative model parameters"""
        if 'synthetic' in self.params['dataset']:
            return
            
        # Transition function
        if self.params['transition_type'] == 'mlp':
            self.p_trans = nn.ModuleList()
            for l in range(self.params['transition_layers']):
                input_dim = self.state_dim if l == 0 else self.params['dim_hidden']*2
                self.p_trans.append(nn.Linear(input_dim, self.params['dim_hidden']*2))
            
            self.p_trans_W_mu = nn.Linear(self.params['dim_hidden']*2, self.state_dim)
            self.p_trans_W_cov = nn.Linear(self.params['dim_hidden']*2, self.state_dim)
            
        elif self.params['transition_type'] == 'simple_gated':
            self.p_gate_embed = nn.Sequential(
                nn.Linear(self.state_dim, self.params['dim_hidden']*2),
                nn.Tanh(),
                nn.Linear(self.params['dim_hidden']*2, self.state_dim)
            )
            
            self.p_z_proj = nn.Sequential(
                nn.Linear(self.state_dim, self.params['dim_hidden']*2),
                nn.Tanh(),
                nn.Linear(self.params['dim_hidden']*2, self.state_dim)
            )
            
            self.p_trans_W_mu = nn.Linear(self.state_dim, self.state_dim)
            with torch.no_grad():
                self.p_trans_W_mu.weight.copy_(torch.eye(self.state_dim))
                self.p_trans_W_mu.bias.zero_()
                
            self.p_trans_W_cov = nn.Linear(self.state_dim, self.state_dim)
        
        # Emission function
        if self.params['emission_type'] == 'mlp':
            self.p_emis = nn.ModuleList()
            for l in range(self.params['emission_layers']):
                input_dim = self.state_dim if l == 0 else self.params['dim_hidden']
                self.p_emis.append(nn.Linear(input_dim, self.params['dim_hidden']))
            
            if self.params['data_type'] == 'binary':
                self.p_emis_out = nn.Linear(self.params['dim_hidden'], self.obs_dim)
            elif self.params['data_type'] == 'real':
                self.p_emis_mu = nn.Linear(self.params['dim_hidden'], self.obs_dim)
                self.p_emis_cov = nn.Linear(self.params['dim_hidden'], self.obs_dim)
    
    def _create_inference_params(self) -> None:
        """Create inference network parameters"""
        # Input embedding
        self.q_input = nn.Linear(self.obs_dim, self.params['rnn_size'])
        
        # LSTM
        if self.params['var_model'] in ['L', 'LR']:
            self.lstm_l = nn.LSTM(
                input_size=self.params['rnn_size'],
                hidden_size=self.params['rnn_size'],
                num_layers=self.params['rnn_layers'],
                batch_first=True
            )
        
        if self.params['var_model'] in ['R', 'LR']:
            self.lstm_r = nn.LSTM(
                input_size=self.params['rnn_size'],
                hidden_size=self.params['rnn_size'],
                num_layers=self.params['rnn_layers'],
                batch_first=True
            )
        
        # Inference model
        if self.params['inference_model'] == 'structured':
            self.q_st = nn.Sequential(
                nn.Linear(self.state_dim, self.params['rnn_size']),
                nn.Tanh()
            )
        
        # Output layers
        self.q_mu = nn.Linear(self.params['rnn_size'], self.state_dim)
        self.q_cov = nn.Linear(self.params['rnn_size'], self.state_dim)
        
        if self.params['var_model'] == 'LR' and self.params['inference_model'] == 'mean_field':
            self.q_mu_r = nn.Linear(self.params['rnn_size'], self.state_dim)
            self.q_cov_r = nn.Linear(self.params['rnn_size'], self.state_dim)
    
    def _get_transition(self, 
                       state: GaussianState,
                       measurement: Optional[torch.Tensor] = None) -> GaussianState:
        """Compute transition p(z_t | z_{t-1})"""
        z = state.mean
        
        if self.params['transition_type'] == 'mlp':
            h = z
            if self.params['use_prev_input'] and measurement is not None:
                X_prev = torch.cat([torch.zeros_like(measurement[:, :1]), measurement[:, :-1]], dim=1)
                h = torch.cat([h, X_prev], dim=-1)
            
            for layer in self.p_trans:
                h = F.tanh(layer(h))
            
            mu = self.p_trans_W_mu(h)
            cov = F.softplus(self.p_trans_W_cov(h))
            
        elif self.params['transition_type'] == 'simple_gated':
            X_prev = None
            if self.params['use_prev_input'] and measurement is not None:
                X_prev = torch.cat([torch.zeros_like(measurement[:, :1]), measurement[:, :-1]], dim=1)
                gate_in = torch.cat([z, X_prev], dim=-1)
            else:
                gate_in = z
                
            gate = torch.sigmoid(self.p_gate_embed(gate_in))
            z_prop = self.p_z_proj(gate_in)
            
            mu = gate * z_prop + (1 - gate) * self.p_trans_W_mu(z)
            cov = F.softplus(self.p_trans_W_cov(F.tanh(z_prop)))
        
        return GaussianState(mu, cov)
    
    def _get_emission(self, state: GaussianState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute emission p(x_t | z_t)"""
        z = state.mean
        
        if self.params['emission_type'] == 'mlp':
            h = z
            for layer in self.p_emis:
                h = F.tanh(layer(h))
            
            if self.params['data_type'] == 'binary':
                return torch.sigmoid(self.p_emis_out(h)), None
            elif self.params['data_type'] == 'real':
                mu = self.p_emis_mu(h)
                cov = F.softplus(self.p_emis_cov(h))
                return mu, cov
    
    def predict(self,
               state: GaussianState,
               measurement: Optional[torch.Tensor] = None) -> GaussianState:
        """
        Predict step of the Deep Kalman Filter.
        Args:
            state: Current state distribution
            measurement: Optional measurement for conditioning
        Returns:
            Predicted state distribution
        """
        return self._get_transition(state, measurement)
    
    def update(self, 
              state: GaussianState, 
              measurement: torch.Tensor) -> GaussianState:
        """
        Исправленный метод update с правильной обработкой измерений
        """
        # measurement shape: (batch_size, obs_dim)
        if measurement.dim() == 2:
            measurement = measurement.unsqueeze(1)  # добавляем временную размерность
            
        # Inference network forward pass
        # print(measurement.shape)
        h = F.tanh(self.q_input(measurement))
        
        if self.params['var_model'] == 'LR':
            h_l, _ = self.lstm_l(h)
            h_r, _ = self.lstm_r(torch.flip(h, [1]))
            h_r = torch.flip(h_r, [1])
            h = h_l + h_r
        elif self.params['var_model'] == 'L':
            h, _ = self.lstm_l(h)
        elif self.params['var_model'] == 'R':
            h, _ = self.lstm_r(h)
        
        mu = self.q_mu(h.squeeze(1))
        cov = F.softplus(self.q_cov(h.squeeze(1)))
        
        return GaussianState(mu, cov)
        
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Исправленный forward pass для обработки последовательностей
        """
        # observations shape: (T, B, obs_dim)
        batch_size = observations.size(1)
        
        # Инициализируем скрытое состояние
        current_state = GaussianState(
            mean=torch.zeros(batch_size, self.state_dim, device=observations.device),
            covariance=torch.eye(self.state_dim, device=observations.device)
                     .unsqueeze(0).repeat(batch_size, 1, 1)
        )
        
        all_means = []
        all_covs = []
        
        # Обрабатываем последовательность по шагам
        for t in range(observations.size(0)):
            # Получаем текущее наблюдение (B, obs_dim)
            measurement = observations[t]
            
            # Predict step
            predicted_state = self.predict(current_state)
            
            # Update step
            updated_state = self.update(predicted_state, measurement)
            
            all_means.append(updated_state.mean)
            all_covs.append(updated_state.covariance)
            
            current_state = updated_state
        
        # Собираем все шаги в тензоры
        return torch.stack(all_means), torch.stack(all_covs)
    
    def loss(self, 
            observations: torch.Tensor, 
            mask: torch.Tensor,
            eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss (negative ELBO)
        Args:
            observations: Input sequence
            mask: Observation mask
            eps: Random noise for reparameterization
        Returns:
            Tuple of (total_loss, nll, kl)
        """
        # Forward pass
        updated_state = self.update(GaussianState(torch.zeros_like(observations[..., :self.state_dim])), 
                                observations)
        predicted_state = self.predict(updated_state, observations)
        
        # Emission parameters
        obs_mu, obs_cov = self._get_emission(updated_state)
        
        # Negative log-likelihood
        if self.params['data_type'] == 'binary':
            nll = F.binary_cross_entropy(obs_mu, observations, reduction='none').sum(-1)
        else:  # real-valued
            std_p = torch.sqrt(obs_cov)
            nll = 0.5 * (np.log(2 * np.pi) + torch.log(obs_cov) + ((observations - obs_mu) / std_p)**2).sum(-1)
        
        nll = (nll * mask).sum()
        
        # KL divergence
        kl = 0.5 * (torch.log(predicted_state.covariance) - torch.log(updated_state.covariance) - 1 + 
                    updated_state.covariance/predicted_state.covariance + 
                    (predicted_state.mean - updated_state.mean)**2/predicted_state.covariance)
        kl = (kl.sum(-1) * mask).sum()
        
        # Total loss
        loss = nll + self.anneal * kl
        
        return loss, nll, kl
    
    def train_step(self, 
                  observations: torch.Tensor,
                  mask: torch.Tensor) -> Tuple[float, float, float]:
        """
        Single training step
        Args:
            observations: Input sequence
            mask: Observation mask
        Returns:
            Tuple of (loss, nll, kl) values
        """
        # Generate noise
        eps = torch.randn_like(observations).repeat(1, 1, self.state_dim)
        
        # Compute loss
        loss, nll, kl = self.loss(observations, mask, eps)
        
        # Backprop
        loss.backward()
        
        # Update annealing
        self.update_ctr += 1
        if self.update_ctr % 1000 == 0:
            self.anneal = min(1.0, 0.01 + self.update_ctr / 1000)
        
        return loss.item(), nll.item(), kl.item()
    
    def infer(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infer latent states
        Args:
            observations: Input sequence
        Returns:
            Tuple of (z, mu, logcov)
        """
        with torch.no_grad():
            eps = torch.randn_like(observations).repeat(1, 1, self.state_dim)
            updated_state = self.update(GaussianState(torch.zeros_like(observations[..., :self.state_dim])), 
                                    observations)
            return (updated_state.mean, 
                    updated_state.mean, 
                    torch.log(F.softplus(updated_state.covariance)))

# kalman/extended.py
import torch
from torch import nn
from typing import Callable, Optional, Tuple
from kalman.filters import BaseFilter
from kalman.gaussian import GaussianState

class ExtendedKalmanFilter(BaseFilter):
    """
    Generic Extended Kalman Filter.

    Parameters
    ----------
    state_dim : int
        Dimension of latent state x.
    obs_dim : int
        Dimension of observation z.
    f : Callable[[torch.Tensor], torch.Tensor]
        Non‑linear transition function x_{k|k} → x_{k+1|k}.
        Must broadcast over batches.  Shape: (*, state_dim) → (*, state_dim)
    h : Callable[[torch.Tensor], torch.Tensor]
        Non‑linear measurement function x_{k|k} → z_pred.
        Shape: (*, state_dim) → (*, obs_dim)
    F_jacobian : Optional[Callable[[torch.Tensor], torch.Tensor]]
        Function returning Jacobian of `f` w.r.t. x.
        Shape: (*, state_dim, state_dim).  If None, computed by autograd.
    H_jacobian : Optional[Callable[[torch.Tensor], torch.Tensor]]
        Function returning Jacobian of `h` w.r.t. x.
        Shape: (*, obs_dim, state_dim).  If None, computed by autograd.
    Q : Optional[torch.Tensor]
        Process‑noise covariance (state_dim × state_dim).  Broadcastable to batch.
        Defaults to identity.
    R : Optional[torch.Tensor]
        Measurement‑noise covariance (obs_dim × obs_dim).  Broadcastable to batch.
        Defaults to identity.
    init_mean : Optional[torch.Tensor]
        Initial state mean (state_dim,) or (B, state_dim).
        If None, zeros.
    init_cov : Optional[torch.Tensor]
        Initial state covariance (state_dim×state_dim) or (B, state_dim, state_dim).
        If None, identity.
    smooth : bool
        Ignored for now (placeholder for RTS smoother).
    eps : float
        Jitter added to diagonals for numerical stability.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        f: Callable[[torch.Tensor], torch.Tensor],
        h: Callable[[torch.Tensor], torch.Tensor],
        F_jacobian: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        H_jacobian: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        Q: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        init_mean: Optional[torch.Tensor] = None,
        init_cov: Optional[torch.Tensor] = None,
        smooth: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__(state_dim, obs_dim, smooth)

        self.f = f
        self.h = h
        self.F_jac_fn = F_jacobian
        self.H_jac_fn = H_jacobian
        self.eps = eps

        eye_x = torch.eye(state_dim)
        eye_z = torch.eye(obs_dim)

        self.register_buffer("Q", eye_x if Q is None else Q)
        self.register_buffer("R", eye_z if R is None else R)

        # Initial posterior (after update of “time 0 – 1”).
        self.register_buffer(
            "_init_mean",
            torch.zeros(state_dim) if init_mean is None else init_mean.clone(),
        )
        self.register_buffer(
            "_init_cov",
            eye_x if init_cov is None else init_cov.clone(),
        )

    # ---------------------------------------------------------------------- #
    #                               helpers                                   #
    # ---------------------------------------------------------------------- #
    def _autograd_jacobian(self, fn: Callable, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian of `fn` wrt x with autograd.
        Shapes
            x  : (*, N)
            out: (*, M, N)
        """
        x = x.clone().requires_grad_(True)
        y = fn(x)  # (*, M)
        # torch.autograd.functional.jacobian explodes memory if we pass batch.
        # We loop over batch dims to stay safe.
        flat_x = x.view(-1, x.shape[-1])
        flat_y = y.view(-1, y.shape[-1])

        J_blocks = [
            torch.autograd.functional.jacobian(
                lambda _xi: fn(_xi.unsqueeze(0)).squeeze(0), xi
            )
            for xi in flat_x
        ]

        J = torch.stack(J_blocks, dim=0)  # (B, M, N)
        return J.view(*x.shape[:-1], *J.shape[-2:])

    def _F(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.F_jac_fn(x)
            if self.F_jac_fn is not None
            else self._autograd_jacobian(self.f, x)
        )

    def _H(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.H_jac_fn(x)
            if self.H_jac_fn is not None
            else self._autograd_jacobian(self.h, x)
        )

    # ---------------------------------------------------------------------- #
    #                      predict / update primitives                       #
    # ---------------------------------------------------------------------- #
    def predict_(
        self,
        state_mean: torch.Tensor,
        state_cov: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EKF time‑update (prediction) step.
        
        Returns:
            Predicted state
        """
        # x̂_{k|k-1}
        m_pred = self.f(state_mean)

        # F_k
        F = self._F(state_mean)  # (*, N, N)

        # P_{k|k-1} = F P Fᵀ + Q
        P_pred = F @ state_cov @ F.transpose(-1, -2) + self.Q

        # add jitter to keep symmetry / positive‑definite.
        P_pred = P_pred + self.eps * torch.eye(self.state_dim, device=P_pred.device)

        return m_pred, P_pred

    def update_(
        self,
        state_mean: torch.Tensor,
        state_cov: torch.Tensor,
        measurement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EKF measurement‑update (correction) step.
        """
        # Innovation y_k = z_k - h(x̂_{k|k-1})
        z_pred = self.h(state_mean)
        y = measurement - z_pred  # (*, obs_dim)

        # H_k
        H = self._H(state_mean)  # (*, obs_dim, state_dim)

        # S_k = H P Hᵀ + R
        S = H @ state_cov @ H.transpose(-1, -2) + self.R
        S = S + self.eps * torch.eye(self.obs_dim, device=S.device)

        # Kalman gain K_k = P Hᵀ S⁻¹
        K = state_cov @ H.transpose(-1, -2) @ torch.linalg.inv(S)

        # Updated mean x̂_{k|k}
        m_upd = state_mean + (K @ y.unsqueeze(-1)).squeeze(-1)

        # Joseph‑form covariance update for numerical stability
        I = torch.eye(self.state_dim, device=state_cov.device)
        ImKH = I - K @ H
        P_upd = ImKH @ state_cov @ ImKH.transpose(-1, -2) + K @ self.R @ K.transpose(
            -1, -2
        )
        P_upd = P_upd + self.eps * torch.eye(self.state_dim, device=P_upd.device)

        return m_upd, P_upd
    
    def predict(self, state: GaussianState) -> GaussianState:
        """
        EKF time‑update (prediction) step.
        
        Returns:
            Predicted state
        """
        m, P = self.predict_(state.mean, state.covariance)
        return GaussianState(m, P)

    def update(self, state: GaussianState, measurement: torch.Tensor) -> GaussianState:
        """
        EKF measurement‑update (correction) step.
        """
        m, P = self.update_(state.mean, state.covariance, measurement)
        return GaussianState(m, P)
 
    # ------------------------------------------------------------------ #
    #                    combined predict‑and‑update step                #
    # ------------------------------------------------------------------ #
    def predict_update(
        self,
        state: GaussianState, 
        measurement: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper that performs a *time‑update* immediately
        followed by a *measurement‑update*.

        Parameters
        ----------
        state : torch.GaussianState
        measurement : torch.Tensor

        Returns
        -------
        new_mean, new_cov : Tuple[torch.Tensor, torch.Tensor]
            Posterior x̂_{k|k} and P_{k|k} after incorporating z_k.
        """
        # 1) predict to time‑k
        predicted_state = self.predict(state)
        # 2) correct with the incoming measurement
        updated_state = self.update(predicted_state, measurement)
        
        return updated_state

    # ---------------------------------------------------------------------- #
    #                       sequence processing (filter)                     #
    # ---------------------------------------------------------------------- #
    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the EKF over a sequence of observations.

        Parameters
        ----------
        observations : torch.Tensor
            Shape (T, B, obs_dim)

        Returns
        -------
        all_states : GaussianState ‑‑ convenient wrapper holding the whole trajectory
        (all_means, all_covs) : Tuple[torch.Tensor, torch.Tensor]
            Means shape (T, B, state_dim)
            Covs  shape (T, B, state_dim, state_dim)
        """
        T, B = observations.shape[:2]
        device = observations.device

        self._init_mean = self._init_mean.to(observations.device)
        self._init_cov = self._init_cov.to(observations.device)

        means = []
        covs = []

        for t in range(T):
            if t > 0:
                self._init_mean, self._init_cov = self.predict_(self._init_mean, self._init_cov)
            self._init_mean, self._init_cov = self.update_(self._init_mean, self._init_cov, observations[t])

            means.append(self._init_mean)
            covs.append(self._init_cov)

        all_means = torch.stack(means, dim=0)  # (T, B, state_dim)
        all_covs = torch.stack(covs, dim=0)    # (T, B, state_dim, state_dim)

        return GaussianState(all_means, all_covs)

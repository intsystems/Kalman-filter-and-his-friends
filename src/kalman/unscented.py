from typing import Callable, Tuple, Optional

import torch
from torch import nn

from kalman.gaussian import GaussianState
from kalman.filters import BaseFilter, SPDParameter


class UnscentedKalmanFilter(BaseFilter):
    r"""
    Scaled-sigma-point Unscented Kalman Filter.

    Parameters
    ----------
    state_dim, obs_dim : int
        Dimensions *n* and *m*.
    f, h : Callable[[torch.Tensor], torch.Tensor]
        Process / measurement models that expect input of shape
        ``(..., n)`` and return output of the same leading shape.
    alpha, beta, kappa : float
        Standard UKF scaling parameters.
    Q, R : torch.Tensor, optional
        Process- and measurement-noise covariances.
    init_mean, init_cov : torch.Tensor, optional
        Initial posterior (after a fictitious step 0 update).
    eps : float
        Jitter added to Cholesky factorizations for numerical stability.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        f: Callable[[torch.Tensor], torch.Tensor],
        h: Callable[[torch.Tensor], torch.Tensor],
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        Q: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        init_mean: Optional[torch.Tensor] = None,
        init_cov: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
    ):
        super().__init__(state_dim, obs_dim)
        self.f, self.h = f, h
        self.eps = eps

        n = state_dim
        lam = alpha**2 * (n + kappa) - n

        # Noise covariances
        eye_x = torch.eye(state_dim, dtype=torch.float64)
        eye_z = torch.eye(obs_dim, dtype=torch.float64)
        Q_val = eye_x if Q is None else Q
        R_val = eye_z if R is None else R

        self._Q_spd = SPDParameter(Q_val.clone())
        self._R_spd = SPDParameter(R_val.clone())

        self._gamma = torch.sqrt(torch.tensor(n + lam, dtype=Q_val.dtype))

        # Weights
        Wm = torch.empty(2 * n + 1, dtype=Q_val.dtype)
        Wc = torch.empty_like(Wm)
        Wm[0] = lam / (n + lam)
        Wc[0] = Wm[0] + (1 - alpha**2 + beta)
        Wm[1:] = 0.5 / (n + lam)
        Wc[1:] = Wm[1:]
        self.register_buffer("Wm", Wm)
        self.register_buffer("Wc", Wc)

        # Initial posterior
        self._init_mean = nn.Parameter(
            torch.zeros(state_dim, dtype=Q_val.dtype) if init_mean is None else init_mean.clone()
        )
        self._init_cov_spd = SPDParameter(
            eye_x.clone() if init_cov is None else init_cov.clone()
        )

        # Cache for sigma points propagated through f (set by predict_, used by update_)
        self._sigmas_f: Optional[torch.Tensor] = None

    @property
    def Q(self) -> torch.Tensor:
        return self._Q_spd()

    @property
    def R(self) -> torch.Tensor:
        return self._R_spd()

    @property
    def _init_cov(self) -> torch.Tensor:
        return self._init_cov_spd()

    # ------------------------------------------------------------------ #
    #                          sigma points                              #
    # ------------------------------------------------------------------ #
    def _sigma_points(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Generate 2n+1 sigma points.
        mean (..., n) -> sigma (..., 2n+1, n)

        Uses the same ordering as filterpy's MerweScaledSigmaPoints:
        [mean, mean + gamma*L[:,0], mean + gamma*L[:,1], ...,
               mean - gamma*L[:,0], mean - gamma*L[:,1], ...]
        """
        n = mean.shape[-1]
        jitter = self.eps * torch.eye(n, device=cov.device, dtype=cov.dtype)
        try:
            chol = torch.linalg.cholesky(cov)
        except RuntimeError:
            chol = torch.linalg.cholesky(cov + jitter)

        scaled = self._gamma * chol  # (..., n, n)
        sigma = [mean]
        for i in range(n):
            sigma.append(mean + scaled[..., :, i])
        for i in range(n):
            sigma.append(mean - scaled[..., :, i])
        return torch.stack(sigma, dim=-2)  # (..., 2n+1, n)

    # ------------------------------------------------------------------ #
    #                       unscented transform                          #
    # ------------------------------------------------------------------ #
    def _unscented_transform(
        self,
        sigma: torch.Tensor,
        noise_cov: torch.Tensor,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sigma (..., 2n+1, n) -> (mean (..., d),
                                  cov  (..., d, d),
                                  y_sigma (..., 2n+1, d))
        """
        y_sigma = fn(sigma)  # (..., 2n+1, d)

        # mean
        y_mean = torch.sum(self.Wm[..., None] * y_sigma, dim=-2)

        # covariance
        y_diff = y_sigma - y_mean.unsqueeze(-2)  # (..., 2n+1, d)
        y_cov = torch.sum(
            self.Wc[..., None, None]
            * y_diff.unsqueeze(-1)
            * y_diff.unsqueeze(-2),
            dim=-3,
        )
        y_cov = y_cov + noise_cov
        return y_mean, y_cov, y_sigma

    # ------------------------------------------------------------------ #
    #                        predict / update                            #
    # ------------------------------------------------------------------ #
    def predict_(self, state_mean: torch.Tensor, state_cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        UKF prediction step.
        Generates sigma points, propagates through f, computes predicted mean and cov.
        Stores propagated sigma points for reuse in update_.
        """
        sigma = self._sigma_points(state_mean, state_cov)
        # Propagate through f
        sigmas_f = self.f(sigma)  # (..., 2n+1, n)
        self._sigmas_f = sigmas_f

        # Predicted mean
        pred_mean = torch.sum(self.Wm[..., None] * sigmas_f, dim=-2)

        # Predicted covariance
        x_diff = sigmas_f - pred_mean.unsqueeze(-2)
        pred_cov = torch.sum(
            self.Wc[..., None, None]
            * x_diff.unsqueeze(-1)
            * x_diff.unsqueeze(-2),
            dim=-3,
        ) + self.Q

        return pred_mean, pred_cov

    def update_(self, state_mean: torch.Tensor, state_cov: torch.Tensor, measurement: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        UKF update step.
        If predict_ was called beforehand, reuses the propagated sigma points (sigmas_f).
        Otherwise, generates fresh sigma points from the current state.
        """
        if self._sigmas_f is not None:
            sigmas_f = self._sigmas_f
            self._sigmas_f = None  # consume
        else:
            # Fallback: generate sigma points from the current state
            sigmas_f = self._sigma_points(state_mean, state_cov)

        # Predicted state mean (recompute from sigmas_f)
        x_mean = torch.sum(self.Wm[..., None] * sigmas_f, dim=-2)

        # Propagate through measurement model
        z_sigma = self.h(sigmas_f)  # (..., 2n+1, m)

        # Predicted measurement mean and covariance
        z_mean = torch.sum(self.Wm[..., None] * z_sigma, dim=-2)
        z_diff = z_sigma - z_mean.unsqueeze(-2)
        z_cov = torch.sum(
            self.Wc[..., None, None]
            * z_diff.unsqueeze(-1)
            * z_diff.unsqueeze(-2),
            dim=-3,
        ) + self.R

        # Cross-covariance P_xz
        x_diff = sigmas_f - x_mean.unsqueeze(-2)  # (..., 2n+1, n)
        P_xz = torch.sum(
            self.Wc[..., None, None]
            * x_diff.unsqueeze(-1)
            * z_diff.unsqueeze(-2),
            dim=-3,
        )  # (..., n, m)

        # Kalman gain
        K = P_xz @ torch.linalg.inv(z_cov)  # (..., n, m)

        # Correction
        y = measurement - z_mean  # (..., m)
        upd_mean = state_mean + (K @ y.unsqueeze(-1)).squeeze(-1)  # (..., n)
        upd_cov = state_cov - K @ z_cov @ K.mT  # (..., n, n)
        return upd_mean, upd_cov

    # ------------------------------------------------------------------ #
    #                   high-level GaussianState API                     #
    # ------------------------------------------------------------------ #
    def predict(self, state: GaussianState) -> GaussianState:
        m, P = self.predict_(state.mean, state.covariance)
        return GaussianState(m, P)

    def update(self, state: GaussianState, measurement: torch.Tensor) -> GaussianState:
        m, P = self.update_(state.mean, state.covariance, measurement)
        return GaussianState(m, P)

    def predict_update(self, state: GaussianState, measurement: torch.Tensor) -> GaussianState:
        return self.update(self.predict(state), measurement)

    # ------------------------------------------------------------------ #
    #                        full sequence pass                          #
    # ------------------------------------------------------------------ #
    def forward(self, observations: torch.Tensor) -> GaussianState:
        """
        Run the UKF over a sequence of observations.

        Parameters
        ----------
        observations : torch.Tensor
            Shape (T, B, obs_dim)

        Returns
        -------
        GaussianState with means (T, B, state_dim) and covs (T, B, state_dim, state_dim)
        """
        T = observations.shape[0]
        mean = self._init_mean.to(observations.device, observations.dtype)
        cov = self._init_cov.to(observations.device, observations.dtype)

        means = []
        covs = []

        for t in range(T):
            if t > 0:
                mean, cov = self.predict_(mean, cov)
            mean, cov = self.update_(mean, cov, observations[t])
            means.append(mean)
            covs.append(cov)

        all_means = torch.stack(means, dim=0)
        all_covs = torch.stack(covs, dim=0)
        return GaussianState(all_means, all_covs)

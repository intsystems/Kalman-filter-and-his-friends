# kalman/unscented.py
from typing import Callable, Tuple, Optional

import torch

from kalman.gaussian import GaussianState          # <-- already in gaussian.py
from kalman.filters import BaseFilter              # <-- your abstract base


class UnscentedKalmanFilter(BaseFilter):
    r"""
    Scaled–sigma‑point Unscented Kalman Filter.

    Parameters
    ----------
    state_dim, obs_dim : int
        Dimensions *n* and *m*.
    f, h : Callable[[torch.Tensor], torch.Tensor]
        Process / measurement models that expect sigma‑points of shape
        ``(..., 2n + 1, n)`` and return the propagated sigma‑points
        with the same shape.
    alpha, beta, kappa : float
        Standard UKF scaling parameters.
    Q, R : torch.Tensor, optional
        Process‑ and measurement‑noise covariances.
    init_mean, init_cov : torch.Tensor, optional
        Initial posterior (after a fictitious step 0 update).
    eps : float
        Jitter added to all Cholesky factorizations and post‑update
        covariances for numerical stability.
    """

    # ------------------------------------------------------------------ #
    #                           constructor                              #
    # ------------------------------------------------------------------ #
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
        eps=1e-7
    ):
        super().__init__(state_dim, obs_dim)
        self.f, self.h = f, h
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)

        n = state_dim
        lam = alpha**2 * (n + kappa) - n
        self._gamma = torch.sqrt(torch.tensor(n + lam, dtype=Q.dtype))
        self.eps = eps
        
        # weights (registered so they automatically follow device / dtype)
        Wm = torch.empty(2 * n + 1, dtype=Q.dtype)
        Wc = torch.empty_like(Wm)
        Wm[0] = lam / (n + lam)
        Wc[0] = Wm[0] + (1 - alpha**2 + beta)
        Wm[1:] = 0.5 / (n + lam)
        Wc[1:] = Wm[1:]
        self.register_buffer('Wm', Wm)
        self.register_buffer('Wc', Wc)

        # noises
        eye_x = torch.eye(state_dim)
        eye_z = torch.eye(obs_dim)
        self.register_buffer("Q", eye_x if Q is None else Q)
        self.register_buffer("R", eye_z if R is None else R)

        # initial posterior
        self.register_buffer("_init_mean", torch.zeros(state_dim) if init_mean is None else init_mean)
        self.register_buffer("_init_cov",  eye_x.clone()     if init_cov  is None else init_cov)

    # ------------------------------------------------ sigma‑points
    def _sigma_points(self, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        mean (..., n) → sigma (..., 2n+1, n)
        """
        n = mean.shape[-1]
        #chol = torch.linalg.cholesky(cov)                     # (..., n, n)
        try:
            chol = torch.linalg.cholesky(cov)                 # (..., n, n)
        except RuntimeError:
            # If Cholesky fails, add small diagonal noise and try again
            eps = self.eps * torch.eye(self.state_dim, device=cov.device, dtype=cov.dtype)
            chol = torch.linalg.cholesky(cov + eps)

        sigma = [mean]                                       # μ
        scaled = self._gamma * chol                          # (..., n, n)
        for i in range(n):
            col = scaled[..., :, i]                          # (..., n)
            sigma.append(mean + col)
            sigma.append(mean - col)
        return torch.stack(sigma, dim=-2)                    # (..., 2n+1, n)

    # ------------------------------------------------ unscented transform
    def _unscented_transform(
        self,
        sigma: torch.Tensor,
        noise_cov: torch.Tensor,
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sigma (..., 2n+1, n) → (mean (..., d),
                                cov  (..., d, d),
                                y_sigma (..., 2n+1, d))
        """
        y_sigma = fn(sigma)                                  # (..., 2n+1, d)

        # mean
        y_mean = torch.sum(self.Wm[..., None] * y_sigma, dim=-2)

        # covariance
        y_diff = y_sigma - y_mean.unsqueeze(-2)              # (..., 2n+1, d)
        y_cov = torch.sum(
            self.Wc[..., None, None]
            * y_diff.unsqueeze(-1) * y_diff.unsqueeze(-2),   # outer prod
            dim=-3,
        )
        y_cov = y_cov + noise_cov
        return y_mean, y_cov, y_sigma

    # ------------------------------------------------ predict / update
    @torch.no_grad()
    def predict_(self, state_mean: torch.Tensor, state_cov: torch.Tensor):
        sigma = self._sigma_points(state_mean, state_cov)
        f = lambda x: self.f(x)                              # expects (..., n) → (..., n)
        pred_mean, pred_cov, _ = self._unscented_transform(sigma, self.Q, f)
        return pred_mean, pred_cov

    @torch.no_grad()
    def update_(self, state_mean: torch.Tensor, state_cov: torch.Tensor, measurement: torch.Tensor):
        # 1. сигма‑точки из априорного распределения
        sigma = self._sigma_points(state_mean, state_cov)

        # 2. прогноз состояния
        f = lambda x: self.f(x)
        x_mean, x_cov, x_sigma = self._unscented_transform(sigma, self.Q, f)

        # 3. прогноз измерения
        h = lambda x: self.h(x)
        z_mean, z_cov, z_sigma = self._unscented_transform(x_sigma, self.R, h)

        # 4. перекрестная ковариация P_xz
        x_diff = x_sigma - x_mean.unsqueeze(-2)              # (..., 2n+1, n)
        z_diff = z_sigma - z_mean.unsqueeze(-2)              # (..., 2n+1, m)
        P_xz = torch.sum(
            self.Wc[..., None, None]
            * x_diff.unsqueeze(-1) * z_diff.unsqueeze(-2),   # (...,2n+1,n,m)
            dim=-3,
        )                                                    # (..., n, m)

        # 5. Калман‑усиление
        K = P_xz @ torch.linalg.inv(z_cov)                   # (..., n, m)

        # 6. корректировка
        y = measurement - z_mean                             # (..., m)
        upd_mean = x_mean + K @ y                            # (..., n)
        upd_cov = x_cov - K @ z_cov @ K.mT                   # (..., n, n)
        return upd_mean, upd_cov
    
    # ================================================================== #
    #                   high‑level GaussianState API                     #
    # ================================================================== #
    def predict(self, state: GaussianState) -> GaussianState:
        m, P = self.predict_(state.mean, state.covariance)
        return GaussianState(m, P)

    def update(self, state: GaussianState, measurement: torch.Tensor) -> GaussianState:
        m, P = self.update_(state.mean, state.covariance, measurement)
        return GaussianState(m, P)

    def predict_update(self, state: GaussianState, measurement: torch.Tensor) -> GaussianState:
        return self.update(self.predict(state), measurement)

    # ================================================================== #
    #                        full sequence pass                          #
    # ================================================================== #
    def forward(self, observations: torch.Tensor):
        """
        Run the UKF over a sequence of observations.

        Parameters
        ----------
        observations : torch.Tensor
            Shape (T, B, obs_dim)

        Returns
        -------
        all_states : GaussianState ‑‑ convenient wrapper holding the whole trajectory
        """
        T, B = observations.shape[:2]
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

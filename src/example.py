"""Example fitlering/smoothing sinusoidal data"""

import argparse
from typing import Tuple

import matplotlib.pyplot as plt
import torch

import kalman.filters as kalman_filters
from kalman.gaussian import GaussianState


def generate_data(n: int, w0: float, noise: float, amplitude: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sinusoidal data:

    x(t) = A sin(w0t)
    z(t) = x(t) + noise * N(0, 1)

    Args:
        n (int): Size of the sequence to generate
        w0 (float): Angular frequency
        noise (float): Gaussian noise standard deviation
        amplitude (float): Amplitude A of the sinus

    Returns:
        torch.Tensor: x(t) state of the system
            Shape: (T, 1, 1)
        torch.Tensor: z(t) measure for each state
            Shape: (T, 1, 1)
    """
    x = amplitude * torch.sin(w0 * torch.arange(n)[..., None, None])
    return x, x + noise * torch.randn_like(x)


def filter_sin(order: int, n: int, measurement_std: float, amplitude: float, nans: bool):
    # Let's do 2 full periods of sinus
    w0 = 4 * torch.pi / n

    # The process std can be computed from the predictions errors made on the last modeled derivative
    # (With sinusoidal functions, both discrete noise modelization gives the same results)
    # Let's assume that the (order+1)-th derivative is nul in expectation. f^(order+1) = N(0, \sigma)
    # Here we know that f^(order+1) = A w0^(order + 1) sin^(order+1)(w0t).
    # We can majore the errors made by Aw0^(order + 1) and could typically choose 5 \sigma = A w0^(order + 1)
    # But empirically as errors are not randomly distributed, this formula
    # puts to much confidence on the process (as it accumulates errors)
    # In practice, we found that using 5 \sigma = Aw0^order worked pretty well (and a sqrt(w0) for order 0)
    process_std = amplitude * w0 ** (order + 0.5 * (order == 0)) / 5
    process_std = max(process_std, 1e-7)  # Prevent floating errors

    print("Parameters")
    print(f"Kalman order: {order}")
    print(f"Measurement noise: {measurement_std}")
    print(f"Process noise: {process_std}")
    print("Data: z(t) = measurement_noise * N(0, 1) + sin(w0 t)")
    print(f"Using w0={w0} for {n} points")

    dim = 1 
    
    state_dim = dim * (order + 1)
    process_matrix = torch.eye(state_dim)

    for i in range(order):
        process_matrix[i, i+1] = 1.0
    
    measurement_matrix = torch.zeros(dim, state_dim)
    measurement_matrix[0, 0] = 1.0
    
    measurement_noise = torch.eye(dim) * measurement_std**2
    
    process_noise = torch.zeros(state_dim, state_dim)
    process_noise[order, order] = process_std**2
    
    kf = kalman_filters.KalmanFilter(
        process_matrix=process_matrix,
        measurement_matrix=measurement_matrix,
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )
    print(kf)
    print()

    x, z = generate_data(n, w0, measurement_std, amplitude)
    if nans:
        z[n // 2 : n // 2 + n // 20] = torch.nan

    initial_covariance = torch.eye(kf.state_dim)
    for i in range(order + 1):
        initial_covariance[i, i] = (amplitude * w0**i / 3)**2
        
    initial_state = GaussianState(
        torch.zeros(kf.state_dim, 1),
        initial_covariance
    )

    states = []
    current_state = initial_state
    
    for t in range(z.shape[0]):
        measurement = z[t]
        if not torch.isnan(measurement).any():
            current_state = kf.update(current_state, measurement)
        
        states.append(current_state)
        
        if t < z.shape[0] - 1:
            current_state = kf.predict(current_state)
    
    means = torch.stack([state.mean for state in states])
    covariances = torch.stack([state.covariance for state in states])
    
    smoothed_means = means
    smoothed_covariances = covariances

    print(f"Filtering MSE: {(means[:, :1] - x).pow(2).mean()}")

    return means, smoothed_means, covariances, smoothed_covariances, x, z

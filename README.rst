.. image:: images/kalman-filter-banner.svg
   :width: 100%
   :align: center
   :alt: Kalman Filter and Extensions

Kalman Filter and Extensions
============================

**Authors**: Matvei Kreinin, Maria Nikitina, Petr Babkin, Anastasia Voznyuk

**Consultant**: Oleg Bakhteev, PhD

Description
-----------

This project focuses on implementing Kalman Filters and their extensions
in a simple and clear manner. Despite their importance, these
state-space models remain underrepresented in the deep learning
community. Our goal is to create a well-documented and efficient
implementation that aligns with existing structured state-space models.

Algorithms Implemented
----------------------

- **Kalman Filter** — linear state estimation
- **Extended Kalman Filter (EKF)** — first-order Taylor linearization
- **Unscented Kalman Filter (UKF)** — sigma-point sampling
- **Variational Bayesian Kalman Filter (VB-AKF)** — adaptive measurement noise estimation
- **Deep Kalman Filter (DKF)** — sequential VAE with neural transition and emission models

Related Work
------------

- `PyTorch implementation of Kalman Filters <https://github.com/raphaelreme/torch-kf>`__
- `Extended Kalman Filter implementation in Pyro <https://pyro.ai/examples/ekf.html>`__
- Compatibility considerations with `S4 and other SSM models <https://github.com/state-spaces/s4>`__

Tech Stack
----------

- **Python**
- **PyTorch** for tensor computation and automatic differentiation
- **NumPy** for numerical computations
- **SciPy** for advanced mathematical functions
- **Jupyter Notebooks** for experimentation and visualization

Quick Start
-----------

.. code-block:: python

   import torch
   from kalman.filters import KalmanFilter

   state_dim, obs_dim = 2, 1
   kf = KalmanFilter(
       transition_matrix=torch.eye(state_dim),
       measurement_matrix=torch.ones(obs_dim, state_dim),
       process_noise=0.01 * torch.eye(state_dim),
       measurement_noise=torch.eye(obs_dim),
       state_dim=state_dim,
       obs_dim=obs_dim,
   )

   state = kf.init_state()
   measurements = torch.randn(10, obs_dim)

   for z in measurements:
       state = kf.predict(state)
       state = kf.update(state, z)
       print("State estimate:", state.mean)

More examples are available in the ``notebooks`` folder.

Links
-----

- `Library Documentation <https://intsystems.github.io/Kalman-filter-and-his-friends/>`__
- `Blogpost <https://www.overleaf.com/read/qyvhbszcygjn#4ff3b8>`__

Authors
-------

- `Matvei Kreinin <https://github.com/kreininmv>`__
- `Maria Nikitina <https://github.com/NikitinaMaria>`__
- `Petr Babkin <https://github.com/petr-parker>`__
- `Anastasia Voznyuk <https://github.com/natriistorm>`__

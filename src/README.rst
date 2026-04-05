Installation
============

Install
-------

.. code-block:: bash

   git clone https://github.com/intsystems/Kalman-filter-and-his-friends
   cd Kalman-filter-and-his-friends
   pip install src/

Uninstall
---------

.. code-block:: bash

   pip uninstall kalman


Basic Usage
===========

1. Kalman Filter
----------------

.. code-block:: python

   import torch
   from kalman.filters import KalmanFilter
   from kalman.gaussian import GaussianState

   kf = KalmanFilter(
       process_matrix=torch.eye(4),
       measurement_matrix=torch.eye(2, 4),
       process_noise=0.01 * torch.eye(4),
       measurement_noise=0.1 * torch.eye(2),
   )

   state = GaussianState(
       mean=torch.zeros(4),
       covariance=torch.eye(4),
   )

   for z in torch.randn(10, 2):
       state = kf.predict(state)
       state = kf.update(state, z)
       print("State:", state.mean)

2. Extended Kalman Filter
-------------------------

.. code-block:: python

   import torch
   from kalman.extended import ExtendedKalmanFilter
   from kalman.gaussian import GaussianState

   ekf = ExtendedKalmanFilter(
       state_dim=4,
       obs_dim=2,
       f=lambda x: x,            # transition function
       h=lambda x: x[..., :2],   # measurement function
       Q=0.01 * torch.eye(4),
       R=0.1 * torch.eye(2),
   )

   state = GaussianState(
       mean=torch.zeros(4),
       covariance=torch.eye(4),
   )

   for z in torch.randn(10, 2):
       state = ekf.predict(state)
       state = ekf.update(state, z)

3. Unscented Kalman Filter
--------------------------

.. code-block:: python

   import torch
   from kalman.unscented import UnscentedKalmanFilter
   from kalman.gaussian import GaussianState

   ukf = UnscentedKalmanFilter(
       state_dim=4,
       obs_dim=2,
       f=lambda x: x,
       h=lambda x: x[..., :2],
       Q=0.01 * torch.eye(4, dtype=torch.float64),
       R=0.1 * torch.eye(2, dtype=torch.float64),
   )

   state = GaussianState(
       mean=torch.zeros(4, dtype=torch.float64),
       covariance=torch.eye(4, dtype=torch.float64),
   )

   for z in torch.randn(10, 2, dtype=torch.float64):
       state = ukf.predict(state)
       state = ukf.update(state, z)

4. Variational Bayesian Kalman Filter
--------------------------------------

.. code-block:: python

   import torch
   from kalman.vkf import VBKalmanFilter

   vbkf = VBKalmanFilter(
       process_matrix=torch.eye(4),
       measurement_matrix=torch.eye(2, 4),
       process_noise=0.01 * torch.eye(4),
       initial_measurement_cov=torch.eye(2),
   )

5. Deep Kalman Filter
---------------------

.. code-block:: python

   from kalman.dkf import DeepKalmanFilter

   dkf = DeepKalmanFilter(
       state_dim=8,
       obs_dim=28,
       params={
           "dataset": "example",
           "transition_type": "mlp",
           "transition_layers": 2,
           "emission_type": "mlp",
           "emission_layers": 2,
           "data_type": "real",
           "dim_hidden": 64,
           "var_model": "LR",
           "rnn_size": 64,
           "rnn_layers": 1,
           "inference_model": "mean_field",
           "use_prev_input": False,
       },
   )

   # Train with ELBO loss
   observations = torch.randn(20, 8, 28)   # (T, B, obs_dim)
   mask = torch.ones(20, 8)
   loss, nll, kl = dkf.loss(observations, mask)

Installation
============

Install
-------

.. code-block:: bash

   git clone https://github.com/intsystems/Kalman-filter-and-his-friends /tmp/Kalman-filter-and-his-friends
   python3 -m pip install /tmp/Kalman-filter-and-his-friends/src/

Uninstall
---------

.. code-block:: bash

   python3 -m pip uninstall kalman


Basic Usage
===========

1. Standard Kalman Filter
-------------------------

.. code-block:: python

   import torch
   from kalman.filters import KalmanFilter
   from kalman.gaussian import GaussianState

   state_dim, obs_dim = 4, 2

   kf = KalmanFilter(
       process_matrix=torch.eye(state_dim),
       measurement_matrix=torch.randn(obs_dim, state_dim),
       process_noise=0.01 * torch.eye(state_dim),
       measurement_noise=torch.eye(obs_dim),
   )

   # Initialize state
   state = GaussianState(
       mean=torch.zeros(state_dim),
       covariance=torch.eye(state_dim),
   )

   measurements = torch.randn(10, obs_dim)

   for z in measurements:
       state = kf.predict(state)
       state = kf.update(state, z)
       print("State estimate:", state.mean)

2. Extended Kalman Filter
-------------------------

.. code-block:: python

   import torch
   from kalman.extended import ExtendedKalmanFilter

   ekf = ExtendedKalmanFilter(
       state_dim=4,
       obs_dim=2,
       f=lambda x: x,          # transition function
       h=lambda x: x[..., :2], # measurement function
   )

   state = ekf.init_state()
   for z in measurements:
       state = ekf.predict(state)
       state = ekf.update(state, z)

3. Unscented Kalman Filter
--------------------------

.. code-block:: python

   from kalman.unscented import UnscentedKalmanFilter

   ukf = UnscentedKalmanFilter(
       state_dim=4,
       obs_dim=2,
       f=lambda x: x,
       h=lambda x: x[..., :2],
   )

4. Variational Bayesian Kalman Filter
--------------------------------------

.. code-block:: python

   from kalman.vkf import VBKalmanFilter

   vbkf = VBKalmanFilter(
       process_matrix=torch.eye(state_dim),
       measurement_matrix=torch.randn(obs_dim, state_dim),
       process_noise=0.01 * torch.eye(state_dim),
       initial_measurement_cov=torch.eye(obs_dim),
   )

5. Deep Kalman Filter
---------------------

.. code-block:: python

   from kalman.dkf import DeepKalmanFilter

   dkf = DeepKalmanFilter(
       state_dim=16,
       obs_dim=4,
       params={
           "transition_type": "mlp",
           "transition_layers": 2,
           "emission_type": "mlp",
           "emission_layers": 2,
           "dim_hidden": 64,
           "var_model": "LR",
           "rnn_size": 64,
           "rnn_layers": 1,
           "inference_model": "structured",
           "use_prev_input": False,
           "data_type": "real",
           "dataset": "example",
       },
   )

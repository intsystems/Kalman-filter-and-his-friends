************
Installation
************

Installing by using PyPi
========================

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
========================

1. Standard Kalman Filter
---------

.. code-block:: python

	from kalman.filters import KalmanFilter
	
	# Define your system matrices
	A = ...  # State transition matrix
	H = ...  # Observation matrix
	Q = ...  # Process noise covariance
	R = ...  # Observation noise covariance
	x0 = ... # Initial state
	P0 = ... # Initial covariance
	
	kf = KalmanFilter(A, H, Q, R, x0, P0)
	
	for z in measurements:
	    kf.predict()
	    kf.update(z)
	    print("Current state estimate:", kf.x)

2. Extended/Unscented/Variational/Deep Kalman Filters
---------
You can use other filters in a similar way:

.. code-block:: python

	from kalman.extended import ExtendedKalmanFilter
	from kalman.unscented import UnscentedKalmanFilter
	from kalman.vkf import VariationalKalmanFilter
	
	# Initialize with your model parameters
	ekf = ExtendedKalmanFilter(...)
	ukf = UnscentedKalmanFilter(...)
	vkf = VariationalKalmanFilter(...)
	dkf = DeepKalmanFilter(...)

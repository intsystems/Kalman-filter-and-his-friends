Unscented Kalman Filter
=======================

UKF avoids computing Jacobians by propagating deterministically chosen sigma points
through the nonlinear functions, then recovering the mean and covariance from the
transformed points.

.. automodule:: kalman.unscented
   :members:
   :undoc-members:
   :show-inheritance:

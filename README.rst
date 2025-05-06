.. image:: images/kalman.png
   :width: 80%
   :align: center

Kalman Filter and Extensions
-------------------------

**Authors**: Matvei Kreinin, Maria Nikitina, Petr Babkin, Anastasia Voznyuk

**Consultant**: Oleg Bakhteev, PhD

💡 Description
--------------

This project focuses on implementing Kalman Filters and their extensions
in a simple and clear manner. Despite their importance, these
state-space models remain underrepresented in the deep learning
community. Our goal is to create a well-documented and efficient
implementation that aligns with existing structured state-space models.

📌 Algorithms Implemented
-------------------------

We plan to implement the following distributions in our library:

- `Kalman Filter`
- `Extended Kalman Filter (EKF)`
- `Unscented Kalman Filter (UKF)`
- `Variational Kalman Filters`

🔗 Related Work
---------------

-  `PyTorch implementation of Kalman
   Filters <https://github.com/raphaelreme/torch-kf?tab=readme-ov-file>`__
-  `Extended Kalman Filter implementation in
   Pyro <https://pyro.ai/examples/ekf.html>`__
-  Compatibility considerations with `S4 and other SSM state-of-the-art
   models <https://github.com/state-spaces/s4>`__

📚 Tech Stack
-------------

The project is implemented using:

-  **Python**
-  **PyTorch** for tensor computation and differentiation
-  **NumPy** for numerical computations
-  **SciPy** for advanced mathematical functions
-  **Jupyter Notebooks** for experimentation and visualization

👨‍💻 Usage
--------

Basic usage examples for different filters will be provided. Below is an
example of using a Kalman Filter in PyTorch:

.. code:: python

   import torch
   from kalman_filter import KalmanFilter

   kf = KalmanFilter(dim_x=4, dim_z=2)
   kf.predict()
   kf.update(torch.tensor([1.0, 2.0]))
   print(kf.x)  # Updated state estimate

More detailed examples and tutorials will be available in the
documentation.

📬 Links
--------

-  `Project Documentation <./docs/plan.md>`__

-  `Project Plan <...>`__

-  .. rubric:: `Matvei Kreinin <https://github.com/kreininmv>`__, `Maria
      Nikitina <https://github.com/NikitinaMaria>`__, `Petr
      Babkin <https://github.com/petr-parker>`__, `Anastasia
      Voznyuk <https://github.com/natriistorm>`__
      :name: matvei-kreinin-maria-nikitina-petr-babkin-anastasia-voznyuk

Feel free to modify and expand this README as needed to fit your
project’s specific goals and implementation details!

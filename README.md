<div align="center">  
    <h1> Kalman Filter and Extensions </h1>
</div>

<div align="center">  
    <img src="doc/source/images/kalman-filter-banner.svg" width="700px" />
</div>

<p align="center">
    <a href="">
        <img alt="Docs" src="https://github.com/intsystems/Kalman-filter-and-his-friends/actions/workflows/docs.yml/badge.svg" />
    </a>
    <a href="">
        <img alt="Blog" src="https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white" />
    </a>
</p>


<table>
    <tr>
        <td align="left"> <b> Title </b> </td>
        <td> Kalman Filter and Extensions </td>
    </tr>
    <tr>
        <td align="left"> <b> Authors </b> </td>
        <td> Matvei Kreinin, Maria Nikitina, Petr Babkin, Anastasia Voznyuk </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Oleg Bakhteev, PhD </td>
    </tr>
</table>

## 💡 Description

This project focuses on implementing Kalman Filters and their extensions in a simple and clear manner. Despite their importance, these state-space models remain underrepresented in the deep learning community. Our goal is to create a well-documented and efficient implementation that aligns with existing structured state-space models.

## 📌 Algorithms Implemented

- [x] **Kalman Filter** — linear state estimation
- [x] **Extended Kalman Filter (EKF)** — first-order Taylor linearization
- [x] **Unscented Kalman Filter (UKF)** — sigma-point sampling
- [x] **Variational Bayesian Kalman Filter (VB-AKF)** — adaptive measurement noise estimation
- [x] **Deep Kalman Filter (DKF)** — sequential VAE with neural transition and emission models

## 🔗 Related Work

- [PyTorch implementation of Kalman Filters](https://github.com/raphaelreme/torch-kf?tab=readme-ov-file)
- [Extended Kalman Filter implementation in Pyro](https://pyro.ai/examples/ekf.html)
- Compatibility considerations with [S4 and other SSM state-of-the-art models](https://github.com/state-spaces/s4)

## 📚 Tech Stack

The project is implemented using:

- **Python**
- **PyTorch** for tensor computation and differentiation
- **NumPy** for numerical computations
- **SciPy** for advanced mathematical functions
- **Jupyter Notebooks** for experimentation and visualization

You can install the required packages using pip:

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/intsystems/Kalman-filter-and-his-friends /tmp/Kalman-filter-and-his-friends
    ```
2. Install the dependencies:
    ```bash
    python3 -m pip install /tmp/Kalman-filter-and-his-friends/src/
    ```

## 👨‍💻 Usage

Basic usage examples for different filters can be found in the `notebooks` folder.

## ✅ Testing

To run the tests after installing the package, execute the following command from the project root:
```bash
PYTHONPATH="${PYTHONPATH}:src" pytest tests/ -v
```

## 📬 Links
- [Library Documentation](https://intsystems.github.io/Kalman-filter-and-his-friends/)
- [Blogpost](https://www.overleaf.com/read/qyvhbszcygjn#4ff3b8)

## Authors
- [Matvei Kreinin](https://github.com/kreininmv)
- [Maria Nikitina](https://github.com/NikitinaMaria)
- [Petr Babkin](https://github.com/petr-parker)
- [Anastasia Voznyuk](https://github.com/natriistorm)

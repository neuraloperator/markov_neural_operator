# Markov Neural Operator (MNO)

This repository contains the code for the paper ["Learning Dissipative Dynamics in Chaotic Systems,"](https://arxiv.org/abs/2106.06898) published in NeurIPS 2022.

In this work, we propose a machine learning framework, which we call the Markov Neural Operator (MNO), to learn the underlying solution operator for dissipative chaotic systems, showing that the resulting learned operator accurately captures short-time trajectories and long-time statistical behavior. Using this framework, we are able to predict various statistics of the invariant measure for the turbulent Kolmogorov Flow dynamics with Reynolds numbers up to 5000.

## Requirements
* Neural operator code is based on the [Fourier Neural Operator (FNO)](https://github.com/zongyi-li/fourier_neural_operator), which requires PyTorch 1.8.0 or later.

## Files
* ``utilities.py``: basic utilities including a reader for .mat files and Sobolev (Hk) and Lp losses.
* ``dissipative_utils.py``: helper functions for encouraging (regularization loss) and enforcing dissipative dynamics (postprocessing).
* ``models/``: model architectures
    * ``densenet.py``: simple feedforward neural network
    *  ``fno_2d.py``: FNO architecture for operators acting on a function space with two spatial dimensions.
* ``data_generation/``: directory containing data generation code for our toy Lorenz-63 dataset and the 1D Kuramoto–Sivashinsky PDE.
* ``scripts/``: scripts for training Lorenz-63 model, 1D KS, and 2D NS equations.
    * ``NS_fno_baseline.py``: FNO baseline trained on 2D NS with Reynolds number 500. No dissipativity or Sobolev loss.
    * ``NS_mno_dissipative.py``: MNO model built on FNO architecture with dissipativity encouraged and Sobolev loss.
    * ``lorenz_densenet.py``: simple feedforward neural network learning Markovian solution operator for Lorenz-63 system. 
    * ``lorenz_dissipative_densenet.py``: simple feedforward neural network with dissipativity encouraged trained on Lorenz-63 system.
* `lorenz.ipynb`: Jupyter notebook with examples to reproduce plots and figures for our Lorenz-63 examples in the paper.
* `visualize_navier_stokes2d.ipynb` : Jupyter notebook with examples to reproduce plots and figures for our 2D Navier-Stokes case study in the paper.

## Datasets
In our work, we train and evaluate on datasets from the Lorenz-63 system (finite-dimensional ODE), Kuramoto–Sivashinsky equation (1D PDE system), and the 2D Navier-Stokes equations (Kolmogorov flow, 2D PDE). Our datasets can be found online under DOI [10.5281/zenodo.74955555](https://zenodo.org/record/7495555).
* Lorenz: Can be found in the `data_generation` directory.
* KS: Can be found in the `data_generation` directory.
* Data generation for 2D Navier-Stokes is based on the data generation scripts in the [FNO repository](https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/navier_stokes).

## Models
In our work, we use three different models to learn the Markovian solution operator. These can be found under the ``models/`` folder in the repository.
* **Lorenz:** Since the Lorenz-63 system is a finite-dimensional ODE system, we use a standard feedforward neural network to learn the Markov solution operator.
* **1D KS and 2D NS equations:** We interpret PDEs as function-space ODEs, and we adopt the 1D and 2D FNO architecture (resp.) to learn the Markov solution operator for the 1D KS and 2D NS equations.

## Citation
```
@inproceedings{MNO,
  title={Learning Dissipative Dynamics in Chaotic Systems},
  author={Zongyi Li and Miguel Liu-Schiaffini and Nikola B. Kovachki and Burigede Liu and Kamyar Azizzadenesheli and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
  year={2022}
}
```

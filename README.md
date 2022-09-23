# Markov Neural Operator (MNO)

This repository contains the code for the paper "Learning Dissipative Dynamics in Chaotic Systems," published in NeurIPS 2022.

In this work, we propose a machine learning framework, which we call the Markov Neural Operator (MNO), to learn the underlying solution operator for dissipative chaotic systems, showing that the resulting learned operator accurately captures short-time trajectories and long-time statistical behavior. Using this framework, we are able to predict various statistics of the invariant measure for the turbulent Kolmogorov Flow dynamics with Reynolds numbers up to 5000.

## Requirements
* Neural operator code is based on the [Fourier Neural Operator (FNO)](https://github.com/zongyi-li/fourier_neural_operator), which requires PyTorch 1.8.0 or later.

## Files
TODO

## Datasets
In our work, we train and evaluate on datasets from the Lorenz-63 system (finite-dimensional ODE), Kuramoto–Sivashinsky equation (1D PDE system), and the 2D Navier-Stokes equations (Kolmogorov flow, 2D PDE).
* Lorenz:
* KS:
* Data generation for 2D Navier-Stokes is based on the data generation scripts in the [FNO repository](https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/navier_stokes).

## Models
In our work, we use three different models to learn the Markovian solution operator. These can be found under the ``models/`` folder in the repository.
* **Lorenz:** Since the Lorenz-63 system is a finite-dimensional ODE system, we use a standard feedforward neural network to learn the Markov solution operator.
* **1D KS and 2D NS equations:** We interpret PDEs as function-space ODEs, and we adopt the 1D and 2D FNO architecture (resp.) to learn the Markov solution operator for the 1D KS and 2D NS equations.

## Citations
TODO

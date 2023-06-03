import numpy as np
import torch

# shape is the tuple shape of each instance
def sample_uniform_spherical_shell(npoints: int, radii: float, shape: tuple):
    ndim = np.prod(shape)
    inner_radius, outer_radius = radii
    pts = []
    for i in range(npoints):
        # uniformly sample radius
        samp_radius = np.random.uniform(inner_radius, outer_radius)
        vec = np.random.randn(ndim) # ref: https://mathworld.wolfram.com/SpherePointPicking.html
        vec /= np.linalg.norm(vec, axis=0)
        pts.append(np.reshape(samp_radius*vec, shape))

    return np.array(pts)

# Partitions of unity - input is real number, output is in interval [0,1]
"""
norm_of_x: real number input
shift: x-coord of 0.5 point in graph of function
scale: larger numbers make a steeper descent at shift x-coord
"""
def sigmoid_partition_unity(norm_of_x, shift, scale):
    return 1/(1 + torch.exp(scale * (norm_of_x - shift)))

# Dissipative functions - input is point x in state space (practically, subset of R^n)
"""
inputs: input point in state space
scale: real number 0 < scale < 1 that scales down input x
"""
def linear_scale_dissipative_target(inputs, scale):
    return scale * inputs

"""
Outputs prediction after post-processing according to:
    rho(|x|) * model(x) + (1 - rho(|x|)) * diss(x)

x: input point as torch tensor
model: torch model
rho: partition of unity, a map from R to [0,1]
diss: baseline dissipative map from R^n to R^n
"""
def part_unity_post_process(x, model, rho, diss):
    return rho(torch.norm(x)) * model(x).reshape(x.shape[0],) + (1 - rho(torch.norm(x))) * diss(x)
import numpy as np

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

def linear_scale_dissipative_target(inputs, scale):
    return scale*inputs

def scale_down_norm(inputs, scale):
    return scale * torch.norm(inputs)
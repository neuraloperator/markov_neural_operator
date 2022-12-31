import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import operator
from functools import reduce

#################################################
#
# Utilities:
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PCA
class PCA(object):
    def __init__(self, x, dim, subtract_mean=True):
        super(PCA, self).__init__()

        # Input size
        x_size = list(x.size())

        # Input data is a matrix
        assert len(x_size) == 2

        # Reducing dimension is less than the minimum of the
        # number of observations and the feature dimension
        assert dim <= min(x_size)

        self.reduced_dim = dim

        if subtract_mean:
            self.x_mean = torch.mean(x, dim=0).view(1, -1)
        else:
            self.x_mean = torch.zeros((x_size[1],), dtype=x.dtype, layout=x.layout, device=x.device)

        # SVD
        U, S, V = torch.svd(x - self.x_mean)
        V = V.t()

        # Flip sign to ensure deterministic output
        max_abs_cols = torch.argmax(torch.abs(U), dim=0)
        signs = torch.sign(U[max_abs_cols, range(U.size()[1])]).view(-1, 1)
        V *= signs

        self.W = V.t()[:, 0:self.reduced_dim]
        self.sing_vals = S.view(-1, )

    def cuda(self):
        self.W = self.W.cuda()
        self.x_mean = self.x_mean.cuda()
        self.sing_vals = self.sing_vals.cuda()

    def encode(self, x):
        return (x - self.x_mean).mm(self.W)

    def decode(self, x):
        return x.mm(self.W.t()) + self.x_mean

    def forward(self, x):
        return self.decode(self.encode(x))

    def __call__(self, x):
        return self.forward(x)


# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True

        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y, std):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if std == True:
            return torch.std(diff_norms / y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms


    def __call__(self, x, y, std=False):
        return self.rel(x, y, std)


class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

    def pval(self, distances, n_permutations=1000):
        r"""Compute a p-value using a permutation test.
        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.MMDStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.
        Returns
        -------
        float
            The estimated p-value."""
        if isinstance(distances, Variable):
            distances = distances.data
        return permutation_test_mat(distances.cpu().numpy(),
                                    self.n_1, self.n_2,
                                    n_permutations,
                                    a00=self.a00, a11=self.a11, a01=self.a01)



#Compute stream function from vorticity (Fourier space)
def stream_function(w, real_space=False):
    device = w.device
    s = w.shape[1]
    w_h = torch.rfft(w, 2, normalized=False, onesided=False)
    psi_h = w_h.clone()

    # Wavenumbers in y and x directions
    k_y = torch.cat((torch.arange(start=0, end=s // 2, step=1, dtype=torch.float32, device=device), \
                          torch.arange(start=-s // 2, end=0, step=1, dtype=torch.float32, device=device)),
                         0).repeat(s, 1)

    k_x = k_y.clone().transpose(0, 1)

    # Negative inverse Laplacian in Fourier space
    inv_lap = (k_x ** 2 + k_y ** 2)
    inv_lap[0, 0] = 1.0
    inv_lap = 1.0 / inv_lap

    #Stream function in Fourier space: solve Poisson equation
    psi_h[...,0] = inv_lap*psi_h[...,0]
    psi_h[...,1] = inv_lap*psi_h[...,1]

    return torch.irfft(psi_h, 2, normalized=False, onesided=False, signal_sizes=(s, s))


#Compute velocity field from stream function (Fourier space)
def velocity_field(stream, real_space=True):
    device = stream.device
    s = stream.shape[1]

    stream_f = torch.rfft(stream, 2, normalized=False, onesided=False)
    # Wavenumbers in y and x directions
    k_y = torch.cat((torch.arange(start=0, end=s // 2, step=1, dtype=torch.float32, device=device), \
                          torch.arange(start=-s // 2, end=0, step=1, dtype=torch.float32, device=device)),
                         0).repeat(s, 1)
    k_x = k_y.clone().transpose(0, 1)

    #Velocity field in x-direction = psi_y
    q_h = stream_f.clone()
    temp = q_h[...,0].clone()
    q_h[...,0] = -k_y*q_h[...,1]
    q_h[...,1] = k_y*temp

    #Velocity field in y-direction = -psi_x
    v_h = stream_f.clone()
    temp = v_h[...,0].clone()
    v_h[...,0] = k_x*v_h[...,1]
    v_h[...,1] = -k_x*temp

    q = torch.irfft(q_h, 2, normalized=False, onesided=False, signal_sizes=(s, s)).squeeze(-1)
    v = torch.irfft(v_h, 2, normalized=False, onesided=False, signal_sizes=(s, s)).squeeze(-1)
    return torch.stack([q,v],dim=3)

def curl3d(u):

    u = u.permute(-1,0,1,2)

    s = u.shape[1]
    kmax = s // 2
    device =u.device

    uh = torch.rfft(u, 3, normalized=False, onesided=False)
    # print(uh.shape)

    xh = uh[1, ..., :]
    yh = uh[0, ..., :]
    zh = uh[2, ..., :]

    k_x = torch.cat((torch.arange(start=0, end=kmax, step=1), torch.arange(start=-kmax, end=0, step=1)), 0).reshape(
        s, 1, 1).repeat(1, s, s).to(device)
    k_y = torch.cat((torch.arange(start=0, end=kmax, step=1), torch.arange(start=-kmax, end=0, step=1)), 0).reshape(
        1, s, 1).repeat(s, 1, s).to(device)
    k_z = torch.cat((torch.arange(start=0, end=kmax, step=1), torch.arange(start=-kmax, end=0, step=1)), 0).reshape(
        1, 1, s).repeat(s, s, 1).to(device)

    xdyh = torch.zeros(xh.shape).to(device)
    xdyh[..., 0] = - k_y * xh[..., 1]
    xdyh[..., 1] = k_y * xh[..., 0]
    xdy = torch.irfft(xdyh, 3, normalized=False, onesided=False)

    xdzh = torch.zeros(xh.shape).to(device)
    xdzh[..., 0] = - k_z * xh[..., 1]
    xdzh[..., 1] = k_z * xh[..., 0]
    xdz = torch.irfft(xdzh, 3, normalized=False, onesided=False)

    ydxh = torch.zeros(xh.shape).to(device)
    ydxh[..., 0] = - k_x * yh[..., 1]
    ydxh[..., 1] = k_x * yh[..., 0]
    ydx = torch.irfft(ydxh, 3, normalized=False, onesided=False)

    ydzh = torch.zeros(xh.shape).to(device)
    ydzh[..., 0] = - k_z * yh[..., 1]
    ydzh[..., 1] = k_z * yh[..., 0]
    ydz = torch.irfft(ydzh, 3, normalized=False, onesided=False)

    zdxh = torch.zeros(xh.shape).to(device)
    zdxh[..., 0] = - k_x * zh[..., 1]
    zdxh[..., 1] = k_x * zh[..., 0]
    zdx = torch.irfft(zdxh, 3, normalized=False, onesided=False)

    zdyh = torch.zeros(xh.shape).to(device)
    zdyh[..., 0] = - k_y * zh[..., 1]
    zdyh[..., 1] = k_y * zh[..., 0]
    zdy = torch.irfft(zdyh, 3, normalized=False, onesided=False)

    w = torch.zeros((s,s,s,3)).to(device)
    w[..., 0] = zdy - ydz
    w[..., 1] = xdz - zdx
    w[..., 2] = ydx - xdy

    return w

def w_to_u(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, -1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.cat([ux, uy], dim=-1)
    return u

def w_to_f(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, 1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    f = torch.fft.irfft2(f_h[:, :, :k_max + 1], dim=[1, 2])
    return f.reshape(batchsize, nx, ny, 1)

def u_to_w(u):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)

    device = u.device
    u = u.reshape(batchsize, nx, ny, 2)
    ux = u[..., 0]
    uy = u[..., 1]

    ux_h = torch.fft.fft2(ux, dim=[1, 2])
    uy_h = torch.fft.fft2(uy, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N)
    # Negative Laplacian in Fourier space
    uxdy_h = 1j * k_y * ux_h
    uydx_h = 1j * k_x * uy_h

    uxdy = torch.fft.irfft2(uxdy_h[:, :, :k_max + 1], dim=[1, 2])
    uydx = torch.fft.irfft2(uydx_h[:, :, :k_max + 1], dim=[1, 2])
    w = uydx - uxdy
    return w

def u_to_f(u):
    return w_to_f(u_to_w(u))

def f_to_u(f):
    batchsize = f.size(0)
    nx = f.size(1)
    ny = f.size(2)

    device = f.device
    f = f.reshape(batchsize, nx, ny, -1)

    f_h = torch.fft.fft2(f, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.stack([ux, uy], dim=-1)
    return u

def f_to_w(f):
    return u_to_w(f_to_u(f))

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

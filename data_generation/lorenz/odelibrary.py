import numpy as np
from matplotlib import pyplot
# from numba import jitclass          # import the decorator
# from numba import boolean, int64, float32, float64    # import the types

import pdb
# Correspondence with Dima via Whatsapp on Feb 24, 2020:
# RK45 (explicit) for slow-system-only
# RK45 (implicit) aka Radau for multi-scale-system
# In both cases, set abstol to 1e-6, reltol to 1e-3, dtmax to 1e-3
# L96spec = [
#     ('K', int64),               # a simple scalar field
#     ('J', int64),               # a simple scalar field
#     ('hx', float64[:]),               # a simple scalar field
#     ('hy', float64),               # a simple scalar field
#     ('F', float64),               # a simple scalar field
#     ('eps', float64),               # a simple scalar field
#     ('k0', float64),               # a simple scalar field
#     ('slow_only', boolean),               # a simple scalar field
#     ('xk_star', float64[:])               # a simple scalar field
# ]

class LDS_COUPLED_X:
  """
  A simple class that implements a coupled linear dynamical system

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    A

  """

  def __init__(_s,
      A = np.array([[0, 1], [-1, 0]]),
      eps_min = 0.001,
      eps_max = 0.05,
      h = 3.0,
      share_gp=True,
      add_closure=False):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    _s.share_gp = share_gp
    _s.A = A
    _s.hx = h # just useful when re-using L96 code
    _s.eps_min = eps_min
    _s.eps_max = eps_max
    _s.K = _s.A.shape[0] # slow state dims
    _s.J = _s.A.shape[0] # fast state dims
    _s.slow_only = False
    _s.exchangeable_states = False
    _s.add_closure = add_closure

  def get_inits(_s):
    state_inits = np.random.uniform(low=-1, high=1, size=_s.K+_s.J)
    # normalize inits so that slow and fast system both start on unit circle
    state_inits[:_s.K] /= np.sqrt(np.sum(state_inits[:_s.K]**2))
    state_inits[_s.K:] /= np.sqrt(np.sum(state_inits[_s.K:]**2))
    return state_inits

  def get_state_names(_s):
    return ['X_'+ str(k+1) for k in range(_s.K)]

  def plot_state_indices(_s):
    return [0, _s.K]

  def slow(_s, x, t):
    ''' Full system RHS '''
    foo_rhs = _s.A @ x
    return foo_rhs

  def eps_f(_s, x):
    return _s.eps_min + 2 * (_s.eps_max - _s.eps_min) * (np.prod(x))**2

  def full(_s, z, t):
    ''' Full system RHS '''
    x = z[:_s.K]
    y = z[_s.K:]
    foo_rhs = np.empty(_s.K + _s.J)
    foo_rhs[:_s.K] = _s.A @ x + _s.hx*y
    foo_rhs[_s.K:] = _s.A @ y / _s.eps_f(x)
    return foo_rhs

  def rhs(_s, z, t):
    if _s.slow_only:
        foo_rhs = _s.slow(z, t)
    else:
        foo_rhs = _s.full(z, t)
    if _s.add_closure:
        foo_rhs += _s.simulate(z)
    return foo_rhs


  def regressed(_s, x, t):
    ''' Only slow variables with RHS learned from data '''
    rhs = _s.rhs(x,t)
    # add data-learned coupling term
    rhs += _s.simulate(x)
    return rhs

  def set_stencil(_s, left = 0, right = 0):
    _s.stencil = np.arange(left, 1 + right)

  def single_step_implied_Ybar(_s, Xnow, Xnext, delta_t):
    # use an euler scheme to back-out the implied avg Ybar_t from X_t and X_t+1
    Ybar = (Xnext - Xnow)/delta_t - _s.rhs(S=Xnow, t=None)

    return Ybar

  def implied_Ybar(_s, X_in, X_out, delta_t):
    # the idea is that X_in are true data coming from a test/training set
    # Xout(k) is the 1-step-ahed prediction associated to Xin(k).
    # In other words Xout(k) = Psi-ML(Xin(k))
    T = X_in.shape[0]
    Ybar = np.zeros( (T, _s.K) )
    for t in range(T):
      Ybar[t,:] = _s.single_step_implied_Ybar(Xnow=X_in[t,:], Xnext=X_out[t,:], delta_t=delta_t)
    return Ybar

  def get_state_limits(_s):
    lims = (None,None)
    return lims

  def set_predictor(_s, predictor):
    _s.predictor = predictor

  # def set_G0_predictor(_s):
  #   _s.predictor = lambda x: _s.hy * x

  def set_null_predictor(_s):
    _s.predictor = lambda x: 0

  def simulate(_s, slow):
    if _s.share_gp:
      return np.reshape(_s.predictor(_s.apply_stencil(slow)), (-1,))
    else:
      return np.reshape(_s.predictor(slow.reshape(1,-1)), (-1,))

  def apply_stencil(_s, slow):
    # behold: the blackest of all black magic!
    # (in a year, I will not understand what this does)
    # the idea: shift xk's so that each row corresponds to the stencil:
    # (x_{k-1}, x_{k}, x_{k+1}), for example,
    # based on '_s.stencil' and 'slow' array (which is (x1,...,xK) )
    return slow[np.add.outer(np.arange(_s.K), _s.stencil) % _s.K]





class LDS_COUPLED:
  """
  A simple class that implements a coupled linear dynamical system

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    A

  """

  def __init__(_s,
      A = np.array([[0, 1], [-1, 0]]),
      eps = 0.05,
      h = 0.1,
      share_gp=True,
      add_closure=False):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    _s.share_gp = share_gp
    _s.A = A
    _s.hx = h # just useful when re-using L96 code
    _s.eps = eps
    _s.K = _s.A.shape[0] # slow state dims
    _s.J = _s.A.shape[0] # fast state dims
    _s.slow_only = False
    _s.exchangeable_states = False
    _s.add_closure = add_closure

  def get_inits(_s):
    state_inits = np.random.uniform(low=-1, high=1, size=_s.K+_s.J)
    # normalize inits so that slow and fast system both start on unit circle
    state_inits[:_s.K] /= np.sqrt(np.sum(state_inits[:_s.K]**2))
    state_inits[_s.K:] /= np.sqrt(np.sum(state_inits[_s.K:]**2))
    return state_inits

  def get_state_names(_s):
    return ['X_'+ str(k+1) for k in range(_s.K)]

  def plot_state_indices(_s):
    return [0, _s.K]

  def slow(_s, x, t):
    ''' Full system RHS '''
    foo_rhs = _s.A @ x
    return foo_rhs

  def full(_s, z, t):
    ''' Full system RHS '''
    x = z[:_s.K]
    y = z[_s.K:]
    foo_rhs = np.empty(_s.K + _s.J)
    foo_rhs[:_s.K] = _s.A @ x + _s.hx*y
    foo_rhs[_s.K:] = _s.A @ y / _s.eps
    return foo_rhs

  def rhs(_s, z, t):
    if _s.slow_only:
        foo_rhs = _s.slow(z, t)
    else:
        foo_rhs = _s.full(z, t)
    if _s.add_closure:
        foo_rhs += _s.simulate(z)
    return foo_rhs


  def regressed(_s, x, t):
    ''' Only slow variables with RHS learned from data '''
    rhs = _s.rhs(x,t)
    # add data-learned coupling term
    rhs += _s.simulate(x)
    return rhs

  def set_stencil(_s, left = 0, right = 0):
    _s.stencil = np.arange(left, 1 + right)

  def single_step_implied_Ybar(_s, Xnow, Xnext, delta_t):
    # use an euler scheme to back-out the implied avg Ybar_t from X_t and X_t+1
    Ybar = (Xnext - Xnow)/delta_t - _s.rhs(S=Xnow, t=None)

    return Ybar

  def implied_Ybar(_s, X_in, X_out, delta_t):
    # the idea is that X_in are true data coming from a test/training set
    # Xout(k) is the 1-step-ahed prediction associated to Xin(k).
    # In other words Xout(k) = Psi-ML(Xin(k))
    T = X_in.shape[0]
    Ybar = np.zeros( (T, _s.K) )
    for t in range(T):
      Ybar[t,:] = _s.single_step_implied_Ybar(Xnow=X_in[t,:], Xnext=X_out[t,:], delta_t=delta_t)
    return Ybar

  def get_state_limits(_s):
    lims = (None,None)
    return lims

  def set_predictor(_s, predictor):
    _s.predictor = predictor

  # def set_G0_predictor(_s):
  #   _s.predictor = lambda x: _s.hy * x

  def set_null_predictor(_s):
    _s.predictor = lambda x: 0

  def simulate(_s, slow):
    if _s.share_gp:
      return np.reshape(_s.predictor(_s.apply_stencil(slow)), (-1,))
    else:
      return np.reshape(_s.predictor(slow.reshape(1,-1)), (-1,))

  def apply_stencil(_s, slow):
    # behold: the blackest of all black magic!
    # (in a year, I will not understand what this does)
    # the idea: shift xk's so that each row corresponds to the stencil:
    # (x_{k-1}, x_{k}, x_{k+1}), for example,
    # based on '_s.stencil' and 'slow' array (which is (x1,...,xK) )
    return slow[np.add.outer(np.arange(_s.K), _s.stencil) % _s.K]



class LDS:
  """
  A simple class that implements a linear dynamical system

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    A

  """

  def __init__(_s,
      A = np.array([[0, 5], [-5, 0]]), share_gp=True, add_closure=False):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    _s.share_gp = share_gp
    _s.A = A
    _s.K = _s.A.shape[0] # state dims
    _s.hx = 1 # just useful when re-using L96 code
    _s.slow_only = False
    _s.exchangeable_states = False
    _s.add_closure = add_closure

  def get_inits(_s):
    state_inits = np.random.randn(_s.K)
    return state_inits

  def get_state_names(_s):
    return ['X_'+ str(k+1) for k in range(_s.K)]

  def plot_state_indices(_s):
    return [0, _s.K]

  def slow(_s, y, t):
    return _s.rhs(y,t)

  def rhs(_s, S, t):
    ''' Full system RHS '''
    foo_rhs = _s.A @ S
    if _s.add_closure:
        foo_rhs += _s.simulate(S)
    return foo_rhs

  def regressed(_s, x, t):
    ''' Only slow variables with RHS learned from data '''
    rhs = _s.rhs(x,t)
    # add data-learned coupling term
    rhs += _s.simulate(x)
    return rhs

  def set_stencil(_s, left = 0, right = 0):
    _s.stencil = np.arange(left, 1 + right)

  def single_step_implied_Ybar(_s, Xnow, Xnext, delta_t):
    # use an euler scheme to back-out the implied avg Ybar_t from X_t and X_t+1
    Ybar = (Xnext - Xnow)/delta_t - _s.rhs(S=Xnow, t=None)

    return Ybar

  def implied_Ybar(_s, X_in, X_out, delta_t):
    # the idea is that X_in are true data coming from a test/training set
    # Xout(k) is the 1-step-ahed prediction associated to Xin(k).
    # In other words Xout(k) = Psi-ML(Xin(k))
    T = X_in.shape[0]
    Ybar = np.zeros( (T, _s.K) )
    for t in range(T):
      Ybar[t,:] = _s.single_step_implied_Ybar(Xnow=X_in[t,:], Xnext=X_out[t,:], delta_t=delta_t)
    return Ybar

  def get_state_limits(_s):
    lims = (None,None)
    return lims

  def set_predictor(_s, predictor):
    _s.predictor = predictor

  # def set_G0_predictor(_s):
  #   _s.predictor = lambda x: _s.hy * x

  def set_null_predictor(_s):
    _s.predictor = lambda x: 0

  def simulate(_s, slow):
    if _s.share_gp:
      return np.reshape(_s.predictor(_s.apply_stencil(slow)), (-1,))
    else:
      return np.reshape(_s.predictor(slow.reshape(1,-1)), (-1,))

  def apply_stencil(_s, slow):
    # behold: the blackest of all black magic!
    # (in a year, I will not understand what this does)
    # the idea: shift xk's so that each row corresponds to the stencil:
    # (x_{k-1}, x_{k}, x_{k+1}), for example,
    # based on '_s.stencil' and 'slow' array (which is (x1,...,xK) )
    return slow[np.add.outer(np.arange(_s.K), _s.stencil) % _s.K]




# @jitclass(L96spec)
class L96M:
  """
  A simple class that implements Lorenz '96M model w/ slow and fast variables

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    K, J, hx, hy, F, eps

  The convention is that the first K variables are slow, while the rest K*J
  variables are fast.
  """

  def __init__(_s,
      K = 9, J = 8, hx = -0.8, hy = 1, F = 10, eps = 2**(-7), k0 = 0, slow_only=False, dima_style=False, share_gp=True, add_closure=False):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    hx = hx * np.ones(K)
    if hx.size != K:
      raise ValueError("'hx' must be a 1D-array of size 'K'")
    _s.predictor = None
    _s.dima_style = dima_style
    _s.share_gp = share_gp # if true, then GP is R->R and is applied to each state independently.
    # if share_gp=False, then GP is R^K -> R^K and is applied to the whole state vector at once.
    _s.slow_only = slow_only
    _s.K = K
    _s.J = J
    _s.hx = hx
    _s.hy = hy
    _s.F = F
    _s.eps = eps
    _s.k0 = k0 # for filtered integration
    _s.exchangeable_states = True
    # 0
    #_s.xk_star = np.random.rand(K) * 15 - 5
    # 1
    #_s.xk_star = np.ones(K) * 5
    # 2
    #_s.xk_star = np.ones(K) * 2
    #_s.xk_star[K//2:] = -0.2
    # 3
    _s.xk_star = 0.0 * np.zeros(K)
    _s.xk_star[0] = 5
    _s.xk_star[1] = 5
    _s.xk_star[-1] = 5
    _s.add_closure = add_closure

  def get_inits(_s, sigma = 15, mu = -5):
    z0 = np.zeros((_s.K + _s.K * _s.J))
    z0[:_s.K] = mu + np.random.rand(_s.K) * sigma
    if _s.slow_only:
      return z0[:_s.K]
    else:
      for k_ in range(_s.K):
        z0[_s.K + k_*_s.J : _s.K + (k_+1)*_s.J] = z0[k_]
      return z0

  def get_state_limits(_s):
    if _s.K==4 and _s.J==4:
      lims = (-27.5, 36.5)
    elif _s.K==9 and _s.J==8:
      lims = (-9.5, 14.5)
    else:
      lims = (None,None)
    return lims

  def get_fast_state_names(_s):
    state_names = []
    for k in range(_s.K):
      state_names += ['Y_' + str(j+1) + ',' + str(k+1) for j in range(_s.J)]
    return state_names

  def get_slow_state_names(_s):
    state_names = ['X_'+ str(k+1) for k in range(_s.K)]
    return state_names

  def get_state_names(_s, get_all=False):
    state_names = _s.get_slow_state_names()
    if get_all or not _s.slow_only:
      state_names += _s.get_fast_state_names()
    return state_names

  def get_fast_state_indices(_s):
    return np.arange(_s.K, _s.K + _s.K * _s.J)

  def plot_state_indices(_s):
    if _s.slow_only:
      return [0, 1, _s.K-1, _s.K-2] # return a 4 coupled slow variables
    else:
      return [0, _s.K] # return 1st slow variable and 1st coupled fast variable

  def set_predictor(_s, predictor):
    _s.predictor = predictor

  def set_G0_predictor(_s):
    _s.predictor = lambda x: _s.hy * x

  def set_null_predictor(_s):
    _s.predictor = lambda x: 0

  def set_stencil(_s, left = 0, right = 0):
    _s.stencil = np.arange(left, 1 + right)

  def hit_value(_s, k, val):
    return lambda t, z: z[k] - val

  def rhs(_s, z, t):
    if _s.slow_only:
        foo_rhs = _s.slow(z, t)
    else:
        foo_rhs = _s.full(z, t)
    if _s.add_closure:
        foo_rhs += _s.simulate(z)
    return foo_rhs

  def full(_s, z, t):
    ''' Full system RHS '''
    K = _s.K
    J = _s.J
    rhs = np.empty(K + K*J)
    x = z[:K]
    y = z[K:]

    ### slow variables subsystem ###
    # compute Yk averages
    Yk = _s.compute_Yk(z)

    # three boundary cases
    rhs[0] =   -x[K-1] * (x[K-2] - x[1]) - x[0]
    rhs[1] =   -x[0]   * (x[K-1] - x[2]) - x[1]
    rhs[K-1] = -x[K-2] * (x[K-3] - x[0]) - x[K-1]

    # general case
    rhs[2:K-1] = -x[1:K-2] * (x[0:K-3] - x[3:K]) - x[2:K-1]

    # add forcing
    rhs[:K] += _s.F

    # add coupling w/ fast variables via averages
    # XXX verify this (twice: sign and vector-vector multiplication)
    rhs[:K] += _s.hx * Yk
    #rhs[:K] -= _s.hx * Yk

    ### fast variables subsystem ###
    # three boundary cases
    rhs[K]  = -y[1]  * (y[2] - y[-1]) - y[0]
    rhs[-2] = -y[-1] * (y[0] - y[-3]) - y[-2]
    rhs[-1] = -y[0]  * (y[1] - y[-2]) - y[-1]

    # general case
    rhs[K+1:-2] = -y[2:-1] * (y[3:] - y[:-3]) - y[1:-2]

    # add coupling w/ slow variables
    for k in range(K):
      rhs[K + k*J : K + (k+1)*J] += _s.hy * x[k]

    # divide by epsilon
    rhs[K:] /= _s.eps

    return rhs

  def decoupled(_s, z, t):
    ''' Only fast variables with fixed slow ones to verify ergodicity '''
    K = _s.K
    J = _s.J
    _i = _s.fidx_dec
    rhs = np.empty(K*J)

    ## boundary: k = 0
    # boundary: j = 0, j = J-2, j = J-1
    rhs[_i(0,0)] = \
        -z[_i(1,0)] * (z[_i(2,0)] - z[_i(J-1,K-1)]) - z[_i(0,0)]
    rhs[_i(J-2,0)] = \
        -z[_i(J-1,0)] * (z[_i(0,1)] - z[_i(J-3,0)]) - z[_i(J-2,0)]
    rhs[_i(J-1,0)] = \
        -z[_i(0,1)] * (z[_i(1,1)] - z[_i(J-2,0)]) - z[_i(J-1,0)]
    # general (for k = 0)
    for j in range(1, J-2):
      rhs[_i(j,0)] = \
          -z[_i(j+1,0)] * (z[_i(j+2,0)] - z[_i(j-1,0)]) - z[_i(j,0)]
    ## boundary: k = 0 (end)

    ## boundary: k = K-1
    # boundary: j = 0, j = J-2, j = J-1
    rhs[_i(0,K-1)] = \
        -z[_i(1,K-1)] * (z[_i(2,K-1)] - z[_i(J-1,K-2)]) - z[_i(0,K-1)]
    rhs[_i(J-2,K-1)] = \
        -z[_i(J-1,K-1)] * (z[_i(0,0)] - z[_i(J-3,K-1)]) - z[_i(J-2,K-1)]
    rhs[_i(J-1,K-1)] = \
        -z[_i(0,0)] * (z[_i(1,0)] - z[_i(J-2,K-1)]) - z[_i(J-1,K-1)]
    # general (for k = K-1)
    for j in range(1, J-2):
      rhs[_i(j,K-1)] = \
          -z[_i(j+1,K-1)] * (z[_i(j+2,K-1)] - z[_i(j-1,K-1)]) - z[_i(j,K-1)]
    ## boundary: k = K-1 (end)

    ## general case for k (w/ corresponding inner boundary conditions)
    for k in range(1, K-1):
      # boundary: j = 0, j = J-2, j = J-1
      rhs[_i(0,k)] = \
          -z[_i(1,k)] * (z[_i(2,k)] - z[_i(J-1,k-1)]) - z[_i(0,k)]
      rhs[_i(J-2,k)] = \
          -z[_i(J-1,k)] * (z[_i(0,k+1)] - z[_i(J-3,k)]) - z[_i(J-2,k)]
      rhs[_i(J-1,k)] = \
          -z[_i(0,k+1)] * (z[_i(1,k+1)] - z[_i(J-2,k)]) - z[_i(J-1,k)]
      # general case for j
      for j in range(1, J-2):
        rhs[_i(j,k)] = \
            -z[_i(j+1,k)] * (z[_i(j+2,k)] - z[_i(j-1,k)]) - z[_i(j,k)]

    ## add coupling w/ slow variables
    for k in range(0, K):
      rhs[k*J : (k+1)*J] += _s.hy * _s.xk_star[k]

    ## divide by epsilon
    rhs /= _s.eps

    return rhs

  def balanced(_s, x, t):
    ''' Only slow variables with balanced RHS '''
    K = _s.K
    rhs = np.empty(K)

    # three boundary cases: k = 0, k = 1, k = K-1
    rhs[0] = -x[K-1] * (x[K-2] - x[1]) - (1 - _s.hx[0]*_s.hy) * x[0]
    rhs[1] = -x[0] * (x[K-1] - x[2]) - (1 - _s.hx[1]*_s.hy) * x[1]
    rhs[K-1] = -x[K-2] * (x[K-3] - x[0]) - (1 - _s.hx[K-1]*_s.hy) * x[K-1]

    # general case
    for k in range(2, K-1):
      rhs[k] = -x[k-1] * (x[k-2] - x[k+1]) - (1 - _s.hx[k]*_s.hy) * x[k]

    # add forcing
    rhs += _s.F

    return rhs

  def slow(_s, x, t):
    ''' Only slow variables with RHS learned from data '''
    K = _s.K
    rhs = np.empty(K)

    # three boundary cases: k = 0, k = 1, k = K-1
    rhs[0] = -x[K-1] * (x[K-2] - x[1]) - x[0]
    rhs[1] = -x[0] * (x[K-1] - x[2]) - x[1]
    rhs[K-1] = -x[K-2] * (x[K-3] - x[0]) - x[K-1]

    # general case
    for k in range(2, K-1):
      rhs[k] = -x[k-1] * (x[k-2] - x[k+1]) - x[k]

    # add forcing
    rhs += _s.F

    return rhs

  def regressed(_s, x, t):
    ''' Only slow variables with RHS learned from data '''
    K = _s.K
    rhs = np.empty(K)

    # three boundary cases: k = 0, k = 1, k = K-1
    rhs[0] = -x[K-1] * (x[K-2] - x[1]) - x[0]
    rhs[1] = -x[0] * (x[K-1] - x[2]) - x[1]
    rhs[K-1] = -x[K-2] * (x[K-3] - x[0]) - x[K-1]

    # general case
    for k in range(2, K-1):
      rhs[k] = -x[k-1] * (x[k-2] - x[k+1]) - x[k]

    # add forcing
    rhs += _s.F

    # add data-learned coupling term
    # XXX verify this (twice: sign and vector-vector multiplication)
    if _s.dima_style:
      rhs += _s.hx * _s.simulate(x)
    else:
      rhs += _s.simulate(x)

    return rhs

  def filtered(_s, t, z):
    ''' Only slow variables with one set of fast ones and RHS learned from data

    Vector z is of size (K + J), i.e. all slow variables + fast variables at k0
    '''
    K = _s.K
    J = _s.J
    rhs = np.empty(K + J)

    ### slow variables subsystem ###
    # compute Yk average for k0
    Yk0 = z[K:].sum() / J

    # three boundary cases: k = 0, k = 1, k = K-1
    rhs[0] = -z[K-1] * (z[K-2] - z[1]) - z[0]
    rhs[1] = -z[0] * (z[K-1] - z[2]) - z[1]
    rhs[K-1] = -z[K-2] * (z[K-3] - z[0]) - z[K-1]

    # general case
    for k in range(2, K-1):
      rhs[k] = -z[k-1] * (z[k-2] - z[k+1]) - z[k]

    # add forcing
    rhs[:K] += _s.F

    # add coupling w/ fast variables via average for k0
    # NOTE This has to be tested; maybe predictor everywhere is better
    rhs[_s.k0] += _s.hx[_s.k0] * Yk0
    # add coupling w/ the rest via simulation
    wo_k0 = np.r_[:_s.k0, _s.k0+1:K]
    Yk_simul = _s.simulate(z[:K])
    rhs[wo_k0] += _s.hx[wo_k0] * Yk_simul[wo_k0]
    #rhs[_s.k0] += _s.hx[_s.k0] * Yk_simul[_s.k0]

    ### fast variables subsystem ###
    # boundary: j = 0, j = J-2, j = J-1
    rhs[K] = -z[K+1] * (z[K+2] - z[-1]) - z[K]
    rhs[K+J-2] = -z[K+J-1] * (z[K] - z[K+J-3]) - z[K+J-2]
    rhs[K+J-1] = -z[K] * (z[K+1] - z[K+J-2]) - z[K+J-1]
    # general case for j
    for j in range(1, J-2):
      rhs[K+j] = -z[K+j+1] * (z[K+j+2] - z[K+j-1]) - z[K+j]

    ## add coupling w/ the k0 slow variable
    rhs[K:] += _s.hy * z[_s.k0]

    ## divide by epsilon
    rhs[K:] /= _s.eps

    return rhs

  def fidx(_s, j, k):
    """Fast-index evaluation (based on the convention, see class description)"""
    return _s.K + k*_s.J + j

  def fidx_dec(_s, j, k):
    """Fast-index evaluation for the decoupled system"""
    return k*_s.J + j

  def simulate(_s, slow):
    if _s.share_gp:
      return np.reshape(_s.predictor(_s.apply_stencil(slow)), (-1,))
    else:
      return np.reshape(_s.predictor(slow.reshape(1,-1)), (-1,))

  def simulate_OLD(_s, slow):
      return np.reshape(_s.predictor(_s.apply_stencil(slow)), (-1,))

  def single_step_implied_Ybar(_s, Xnow, Xnext, delta_t):
    # use an euler scheme to back-out the implied avg Ybar_t from X_t and X_t+1
    Ybar = (Xnext - Xnow)/delta_t - _s.slow(x=Xnow, t=None)

    # divide by hx
    Ybar /= _s.hx

    return Ybar

  def implied_Ybar(_s, X_in, X_out, delta_t):
    # the idea is that X_in are true data coming from a test/training set
    # Xout(k) is the 1-step-ahed prediction associated to Xin(k).
    # In other words Xout(k) = Psi-ML(Xin(k))
    T = X_in.shape[0]
    Ybar = np.zeros( (T, _s.K) )
    for t in range(T):
      Ybar[t,:] = _s.single_step_implied_Ybar(Xnow=X_in[t,:], Xnext=X_out[t,:], delta_t=delta_t)
    return Ybar

  def compute_Yk(_s, z):
    return z[_s.K:].reshape( (_s.J, _s.K), order = 'F').sum(axis = 0) / _s.J
    # TODO delete these two lines after testing
    #_s.Yk = z[_s.K:].reshape( (_s.J, _s.K), order = 'F').sum(axis = 0)
    #_s.Yk /= _s.J

  def gather_pairs(_s, tseries):
    n = tseries.shape[1]
    pairs = np.empty( (_s.K * n, _s.stencil.size + 1) )
    for j in range(n):
      pairs[_s.K * j : _s.K * (j+1), :-1] = _s.apply_stencil(tseries[:_s.K, j])
      pairs[_s.K * j : _s.K * (j+1), -1] = _s.compute_Yk(tseries[:,j])
    return pairs

  def gather_pairs_k0(_s, tseries):
    n = tseries.shape[1]
    pairs = np.empty( (n, 2) )
    for j in range(n):
      pairs[j, 0] = tseries[_s.k0, j]
      pairs[j, 1] = tseries[_s.K:, j].sum() / _s.J
    return pairs

  def apply_stencil(_s, slow):
    # behold: the blackest of all black magic!
    # (in a year, I will not understand what this does)
    # the idea: shift xk's so that each row corresponds to the stencil:
    # (x_{k-1}, x_{k}, x_{k+1}), for example,
    # based on '_s.stencil' and 'slow' array (which is (x1,...,xK) )
    return slow[np.add.outer(np.arange(_s.K), _s.stencil) % _s.K]

################################################################################
# end of L96M ##################################################################
################################################################################
# L63spec = [
#     ('a', float32),               # a simple scalar field
#     ('b', float32),               # a simple scalar field
#     ('c', float32),               # a simple scalar field
# ]

# @jitclass(L63spec)
class L63:
  """
  A simple class that implements Lorenz 63 model

  The class computes RHS's to make use of scipy's ODE solvers.

  Parameters:
    a, b, c

  """

  def __init__(_s,
      a = 10, b = 28, c = 8/3, share_gp=True, add_closure=False):
    '''
    Initialize an instance: setting parameters and xkstar
    '''
    _s.share_gp = share_gp
    _s.a = a
    _s.b = b
    _s.c = c
    _s.K = 3 # state dims
    _s.hx = 1 # just useful when re-using L96 code
    _s.slow_only = False
    _s.exchangeable_states = False
    _s.add_closure = add_closure

  def get_inits(_s):
    (xmin, xmax) = (-10,10)
    (ymin, ymax) = (-20,30)
    (zmin, zmax) = (10,40)

    xrand = xmin+(xmax-xmin)*np.random.random()
    yrand = ymin+(ymax-ymin)*np.random.random()
    zrand = zmin+(zmax-zmin)*np.random.random()
    state_inits = np.array([xrand, yrand, zrand])
    return state_inits

  def get_state_names(_s):
    return ['x','y','z']

  def plot_state_indices(_s):
    return [0,1,2]

  def slow(_s, y, t):
    return _s.rhs(y,t)

  def rhs(_s, S, t):
    ''' Full system RHS '''
    a = _s.a
    b = _s.b
    c = _s.c
    x = S[0]
    y = S[1]
    z = S[2]

    foo_rhs = np.empty(3)
    foo_rhs[0] = -a*x + a*y
    foo_rhs[1] = b*x - y - x*z
    foo_rhs[2] = -c*z + x*y

    if _s.add_closure:
        foo_rhs += _s.simulate(S)
    return foo_rhs

  def regressed(_s, x, t):
    ''' Only slow variables with RHS learned from data '''
    rhs = _s.rhs(x,t)
    # add data-learned coupling term
    rhs += _s.simulate(x)
    return rhs

  def set_stencil(_s, left = 0, right = 0):
    _s.stencil = np.arange(left, 1 + right)

  def single_step_implied_Ybar(_s, Xnow, Xnext, delta_t):
    # use an euler scheme to back-out the implied avg Ybar_t from X_t and X_t+1
    Ybar = (Xnext - Xnow)/delta_t - _s.rhs(S=Xnow, t=None)

    return Ybar

  def implied_Ybar(_s, X_in, X_out, delta_t):
    # the idea is that X_in are true data coming from a test/training set
    # Xout(k) is the 1-step-ahed prediction associated to Xin(k).
    # In other words Xout(k) = Psi-ML(Xin(k))
    T = X_in.shape[0]
    Ybar = np.zeros( (T, _s.K) )
    for t in range(T):
      Ybar[t,:] = _s.single_step_implied_Ybar(Xnow=X_in[t,:], Xnext=X_out[t,:], delta_t=delta_t)
    return Ybar

  def get_state_limits(_s):
    lims = (None,None)
    return lims

  def set_predictor(_s, predictor):
    _s.predictor = predictor

  # def set_G0_predictor(_s):
  #   _s.predictor = lambda x: _s.hy * x

  def set_null_predictor(_s):
    _s.predictor = lambda x: 0

  def simulate(_s, slow):
    if _s.share_gp:
      return np.reshape(_s.predictor(_s.apply_stencil(slow)), (-1,))
    else:
      return np.reshape(_s.predictor(slow.reshape(1,-1)), (-1,))

  def apply_stencil(_s, slow):
    # behold: the blackest of all black magic!
    # (in a year, I will not understand what this does)
    # the idea: shift xk's so that each row corresponds to the stencil:
    # (x_{k-1}, x_{k}, x_{k+1}), for example,
    # based on '_s.stencil' and 'slow' array (which is (x1,...,xK) )
    return slow[np.add.outer(np.arange(_s.K), _s.stencil) % _s.K]

################################################################################
# end of L63 ##################################################################
################################################################################

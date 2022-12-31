
#!/usr/bin/env python
# # -*- coding: utf-8 -*-
import pdb
import numpy as np
import pickle
from time import time
from scipy.integrate import solve_ivp
from odelibrary import L63

import matplotlib
import matplotlib.pyplot as plt

# read in ODE class
l63 = L63()

# swap input order for expectation of scipy.integrate.solve_ivp
f_ode = lambda t, y: l63.rhs(y, t)

T1 = 100; T2 = 100; dt = 0.001;

# INTEGRATION
u0 = l63.get_inits()
t0 = 0

print("Integrating through an initial transient phase to reach the attractor...")
tstart = time()
t_span = [t0, T1]
t_eval = np.array([t0+T1])
sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=dt, method='RK45')

print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')

print("Integrating trajectory on the attractor...")
tstart = time()
u0 = np.squeeze(sol.y)
t_span = [t0, T2]
t_eval_tmp = np.arange(t0, T2, dt)
t_eval = np.zeros(len(t_eval_tmp)+1)
t_eval[:-1] = t_eval_tmp
t_eval[-1] = T2
sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, max_step=dt, method='RK45')
u = sol.y.T

data = {
    "T1":T1,
    "T2":T2,
    "dt":dt,
    "u":u,
}

print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')

# save data
with open("data.pickle", "wb") as file:
    # Pickle the "data" dictionary using the highest protocol available.
    pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

# plot trajectory
T_plot = 20
n_plot = int(T_plot/dt)
K = u.shape[1] #number of ode states
fig, axes = plt.subplots(nrows=K, ncols=1,figsize=(12, 6))
times = dt*np.arange(n_plot)
pdb.set_trace()
for k in range(K):
    axes[k].plot(times, u[:n_plot,k], linewidth=2)
    axes[k].set_ylabel('X_{k}'.format(k=k))
axes[k].set_xlabel('Time')
fig.suptitle('Lorenz 63 Trajectory simulated with RK45')
plt.savefig('l63trajectory')
plt.close()

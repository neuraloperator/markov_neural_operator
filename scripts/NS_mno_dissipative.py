import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../')
from utilities import *
from dissipative_utils import sample_uniform_spherical_shell, linear_scale_dissipative_target

sys.path.append('../models')
from fno_2d import *

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

# Main
ntrain = 900
ntest = 100

S = 64

# DISSIPATIVE REGULARIZATION PARAMETERS
# below, the number before multiplication by S is the radius in the L2 norm of the function space
radius = 156.25 * S # radius of inner ball
scale_down = 0.5 # rate at which to linearly scale down inputs
loss_weight = 0.01 * (S**2) # normalized by L2 norm in function space
radii = (radius, (525 * S) + radius) # inner and outer radii, in L2 norm of function space
sampling_fn = sample_uniform_spherical_shell #numsampled is batch size
target_fn = linear_scale_dissipative_target

modes = 20
width = 64

in_dim = 1
out_dim = 1

batch_size = 50

epochs = 50
learning_rate = 0.0005
scheduler_step = 10
scheduler_gamma = 0.5

loss_k = 1
loss_group = True

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'NS_fourier_MNO_dissipative_N_'+str(ntrain)+'_k' + str(loss_k)+'_g' + str(loss_group)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path

sub = 1 # spatial subsample
S = 64

T_in = 100 # skip first 100 seconds of each trajectory to let trajectory reach attractor
T = 400 # seconds to extract from each trajectory in data
T_out = T_in + T
step = 1 # Seconds to learn solution operator

t1 = default_timer()
data = np.load('../data/KFvorticity_Re500_N1000_T500.npy')
data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]

train_a = data[:ntrain,T_in-1:T_out-1].reshape(ntrain*T, S, S)
train_u = data[:ntrain,T_in:T_out].reshape(ntrain*T, S, S)

test_a = data[-ntest:,T_in-1:T_out-1].reshape(ntest*T, S, S)
test_u = data[-ntest:,T_in:T_out].reshape(ntest*T, S, S)

assert (S == train_u.shape[2])

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

# Model
model = Net2d(in_dim, out_dim, S, modes, width).cuda()
print(model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

lploss = LpLoss(size_average=False)
h1loss = HsLoss(k=1, group=False, size_average=False)
h2loss = HsLoss(k=2, group=False, size_average=False)
myloss = HsLoss(k=loss_k, group=loss_group, size_average=False)
dissloss = nn.MSELoss(reduction='mean')

# Training
for ep in range(1, epochs + 1):
    model.train()
    t1 = default_timer()
    train_loss = 0
    diss_l2 = 0
    for x, y in train_loader:
        x = x.to(device).view(batch_size, S, S, in_dim)
        y = y.to(device).view(batch_size, S, S, out_dim)

        out = model(x).reshape(batch_size, S, S, out_dim)
        data_loss = myloss(out, y)
        train_loss += data_loss.item()

        x_diss = torch.tensor(sampling_fn(x.shape[0], radii, (S, S, 1)), dtype=torch.float).to(device)
        assert(x_diss.shape == x.shape)
        y_diss = torch.tensor(target_fn(x_diss, scale_down), dtype=torch.float).to(device)
        out_diss = model(x_diss).reshape(-1, out_dim)
        diss_loss = (1/(S**2)) * loss_weight * dissloss(out_diss, y_diss.reshape(-1, out_dim)) # weighted by 1 / (S**2)
        diss_l2 += diss_loss.item()

        loss = data_loss + diss_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2 = 0
    test_h1 = 0
    test_h2 = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).view(batch_size, S, S, in_dim)
            y = y.to(device).view(batch_size, S, S, out_dim)

            out = model(x).reshape(batch_size, S, S, out_dim)
            test_l2 += lploss(out, y).item()
            test_h1 += h1loss(out, y).item()
            test_h2 += h2loss(out, y).item()

    t2 = default_timer()
    scheduler.step()
    print("Epoch " + str(ep) + " completed in " + "{0:.{1}f}".format(t2-t1, 3) + " seconds. Train err:", "{0:.{1}f}".format(train_loss/(ntrain*T), 3), "Test L2 err:", "{0:.{1}f}".format(test_l2/(ntest*T), 3), "Test H1 err:",  "{0:.{1}f}".format(test_h1/(ntest*T), 3), "Test H2 err:",  "{0:.{1}f}".format(test_h2/(ntest*T), 3), "Train diss err:", "{0:.{1}f}".format(diss_l2/(ntrain), 3))
    print(ep, t2 - t1, train_loss/(ntrain*T), test_l2/(ntest*T), test_h1/(ntest*T), test_h2/(ntest*T), diss_l2/(ntrain))

torch.save(model, path_model)
print("Weights saved to", path_model)

# Long-time prediction
model.eval()
test_a = test_a[0,:,:]

T = 10000
pred = torch.zeros(S,S,T)
out = test_a.reshape(1,S,S).cuda()
with torch.no_grad():
    for i in range(T):
        out = model(out.reshape(1,S,S,in_dim))
        pred[:,:,i] = out.view(S,S)

scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})
print("10000 seconds of predictions saved to", pred_path)
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from timeit import default_timer
import scipy.io

import sys
sys.path.append('../')
from utilities import *
from dissipative_utils import sample_uniform_spherical_shell, linear_scale_dissipative_target

sys.path.append('../models')
from densenet import *

torch.manual_seed(0)
np.random.seed(0)

# Main
ntrain = 160000
ntest = 38000
scale_inputs = False

# DISSIPATIVE REGULARIZATION PARAMETERS
scale_down = 0.5 # rate at which to linearly scale down inputs
radius = 90 # radius of inner ball
loss_weight = 1 # weighting term between data loss and dissipativity term
radii = (radius, radius + 40) # inner and outer radii
sampling_fn = sample_uniform_spherical_shell # numsamples is batch size
target_fn = linear_scale_dissipative_target

in_dim = 3
out_dim = 3

batch_size = 256
epochs = 1000
learning_rate = 0.0005

layers = [in_dim, in_dim*50, in_dim*50,  in_dim*50, in_dim*50, in_dim*50, in_dim*50, out_dim]
nonlinearity = nn.ReLU

rel_loss = True # relative Lp loss

scheduler_step = 100
scheduler_gamma = 0.5

print()
print("Epochs:", epochs)
print("Learning rate:", learning_rate)
print("Scheduler step:", scheduler_step)
print("Scheduler gamma:", scheduler_gamma)
print()

path = 'lorenz_dissipative_densenet_dt_0_05_inner_rad'+str(int(radius))+'_outer_rad'+str(int(radii[1]))+'_lambda'+str(scale_down).replace('.','_')+'_diss_weight'+str(loss_weight).replace('.','_')+'_time'+str(ntrain)+'_ep' + str(epochs) + '_lr' + str(learning_rate).replace('.','_') + '_schedstep' + str(scheduler_step).replace('.','_') + '_relLp' + str(rel_loss) + '_layers' + str(layers)[1:-1].replace(', ', '_')
path_model = 'model/'+path
print(path)

# Data
sub = 6 # temporal subsampling rate
steps_per_sec = 21 # given temporal subsampling, num of time-steps per second
t1 = default_timer()

predloader = MatReader('../data/L63T10000.mat')
data = predloader.read_field('u')[::sub]
data = torch.tensor(data, dtype=torch.float)

train_a = data[:ntrain]
train_u = data[1:ntrain+1]

train_mean = torch.mean(train_a)
train_max = torch.max(train_a)
train_min = torch.min(train_a)

if scale_inputs:
    train_a = (train_a - train_mean)/(train_max - train_min)
    train_u = (train_u - train_mean)/(train_max - train_min)

test_a = data[-ntest:-1]
test_u = data[-ntest + 1:]

if scale_inputs:
    test_a = (test_a - train_mean)/(train_max - train_min)
    test_u = (test_u - train_mean)/(train_max - train_min)

assert train_a.shape == train_u.shape
assert test_a.shape == test_u.shape

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
print()
device = torch.device('cuda')

# Model
model = DenseNet(layers, nonlinearity).cuda()
print("Model parameters:", model.count_params())
print()

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

if rel_loss:
    trainloss = LpLoss(size_average=False)
    dissloss = LpLoss(size_average=False)
    testloss = LpLoss(size_average=False)
    test_dissloss = LpLoss(size_average=False)
    testloss_1sec = LpLoss(size_average=False)
else:
    trainloss = nn.MSELoss(reduction='sum')
    dissloss = nn.MSELoss(reduction='sum')
    testloss = nn.MSELoss(reduction='sum')
    test_dissloss = nn.MSELoss(reduction='sum')
    testloss_1sec = nn.MSELoss(reduction='sum')

for ep in range(1, epochs + 1):
    model.train()
    t1 = default_timer()
    one_sec_count = 0
    train_l2 = 0
    diss_l2 = 0
    for x, y in train_loader:
        x = x.to(device).view(-1, out_dim)
        y = y.to(device).view(-1, out_dim)

        out = model(x).reshape(-1, out_dim)
        data_loss = trainloss(out, y)
        train_l2 += data_loss.item()

        x_diss = torch.tensor(sampling_fn(x.shape[0], radii, (in_dim,)), dtype=torch.float).to(device)
        assert(x_diss.shape == x.shape)
        y_diss = torch.tensor(target_fn(x_diss, scale_down), dtype=torch.float).to(device)
        out_diss = model(x_diss).reshape(-1, out_dim)
        diss_loss = loss_weight*dissloss(out_diss, y_diss) # weighted
        diss_l2 += diss_loss.item()

        loss = data_loss + diss_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2 = 0
    test_diss_l2 = 0
    test_l2_1_sec = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).view(-1, out_dim)
            y = y.to(device).view(-1, out_dim)

            out = model(x).reshape(-1, out_dim)
            test_l2 += testloss(out, y).item()

            x_diss = torch.tensor(sampling_fn(x.shape[0], radii, (in_dim,)), dtype=torch.float).to(device)
            assert(x_diss.shape == x.shape)
            y_diss = torch.tensor(target_fn(x_diss, scale_down), dtype=torch.float).to(device)
            out_diss = model(x_diss).reshape(-1, out_dim)
            test_diss_loss = test_dissloss(out_diss, y_diss) # unweighted
            test_diss_l2 += test_diss_loss.item()

            x_subsample = x[::steps_per_sec]
            x_1sec = x_subsample[:-2] # inputs
            y_1sec = x_subsample[1:-1] # ground truth
            out = x_1sec
            for i in range(steps_per_sec):
                out = model(out).reshape(-1, out_dim)
            test_1_sec_loss = testloss_1sec(out, y_1sec)
            test_l2_1_sec += test_1_sec_loss.item()
            one_sec_count += (int)(y_1sec.shape[0])

    t2 = default_timer()
    scheduler.step()
    print("Epoch " + str(ep) + " completed in " + "{0:.{1}f}".format(t2-t1, 3) + " seconds. Train L2 err:", "{0:.{1}f}".format(train_l2/(ntrain), 3), "Test L2 err:", "{0:.{1}f}".format(test_l2/(ntest), 3), "Train diss. err:",  "{0:.{1}f}".format(diss_l2/(ntrain), 3), "Test diss. err:",  "{0:.{1}f}".format(test_diss_l2/(ntest), 3), "Test L2 err over 1 sec:", "{0:.{1}f}".format(test_l2_1_sec/(one_sec_count), 3))

torch.save(model, path_model)
print("Weights saved to", path_model)

# Long-time prediction
model.eval()
test_a = test_a[0]

T = 10000 * steps_per_sec
pred = torch.zeros(T, out_dim)
out = test_a.reshape(1,in_dim).cuda()
with torch.no_grad():
    for i in range(T):
        out = model(out.reshape(1,in_dim))
        pred[i] = out.view(out_dim,)

scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})
print("10000 seconds of predictions saved to", 'pred/'+path+'.mat')
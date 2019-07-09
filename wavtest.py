import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import dataloader as dl
import model as md
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


#valset = dl.dataLoader('data/Val_set', 8000)


# val_loader = torch.utils.data.DataLoader(valset, batch_size=100,
#                                         shuffle=True, num_workers=1,
#                                         pin_memory=True)

Unet = md.CNN(1440)
Unet.load_state_dict(torch.load('model/CNN_aug_60s.w', map_location='cpu'))
Unet.eval()

# Unet.eval()
#pred_val = []
# for i, (frame) in enumerate(val_loader):
##    x = frame['intensity'].type(torch.FloatTensor).squeeze(1)
#    y = frame['label'].type(torch.FloatTensor)
#    uinput = Variable(x)
#    utarget = Variable(y)
#    pred_data = torch.sigmoid(Unet(uinput.unsqueeze(1))).squeeze(1)
#    e = (pred_data > 0.5 - utarget).detach().numpy()
#    pred_val.append(e)
#pred_val = np.stack(pred_val)
#false_alarm = np.sum(pred_val == -1) / 200
#miss = np.sum(pred_val == 1) / 6000
# Unet.train()

'''
data = sio.loadmat('data/test_grid_dec_60s')
dd = data['test_all']

results = np.zeros((dd.shape[0], dd.shape[1]))
for i in range(dd.shape[0]):
    x = torch.from_numpy(dd[i, :, :]).float().unsqueeze(1)
    uinput = Variable(x)
    pred_data = torch.sigmoid(Unet(uinput)).squeeze(1)
    results[i, :] = pred_data.detach().numpy()

fig = plt.figure()
x = np.arange(-5, 5.5, 0.5)
y = np.arange(-15, 15.5, 0.5)
X, Y = np.meshgrid(x, y)
ax = fig.gca(projection='3d')
p = ax.plot_surface(X, Y, results.T, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
cb = fig.colorbar(p)
plt.show()

fig, ax = plt.subplots()
p = ax.pcolor(X, Y, results.T, cmap=cm.RdBu, vmin=abs(results).min(), vmax=abs(results).max())
cb = fig.colorbar(p)
plt.show()

sio.savemat('result_light_dec_60s', {'results_light': results})

import pdb
pdb.set_trace()  # breakpoint 38b530b9 //
'''

DISPLAY = 0

for n in range(379):
    #n = 26
    data = sio.loadmat('data/Test_set_60s_30/' + str(n + 1))
    dd = data['test_all']
    results = np.zeros((dd.shape[0], dd.shape[1]))
    for i in range(dd.shape[0]):
        x = torch.from_numpy(dd[i, :, :]).float().unsqueeze(1)
        uinput = Variable(x)
        pred_data = torch.sigmoid(Unet(uinput)).squeeze(1)
        results[i, :] = pred_data.detach().numpy()

    if DISPLAY == 1:
        fig = plt.figure()
        x = np.arange(-5, 5.5, 0.5)
        y = np.arange(-30, 30.5, 0.5)
        X, Y = np.meshgrid(x, y)
        ax = fig.gca(projection='3d')
        p = ax.plot_surface(X, Y, results.T, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        cb = fig.colorbar(p)
        plt.show()

        fig, ax = plt.subplots()
        p = ax.pcolor(X, Y, results.T, cmap=cm.RdBu, vmin=abs(results).min(), vmax=abs(results).max())
        cb = fig.colorbar(p)
        plt.show()

    path = 'data/test_results_aug_60s_30/' + str(n + 1)
    sio.savemat(path, {'results': results})

    #sio.savemat(path, {'results': results})
    print(n)
#results = results / np.max(results)
#xx, yy = np.where(results > 0.9)
#zz = results[results > 0.9]

#print(np.sum(x[xx] * zz / sum(zz)))
#print(np.sum(y[yy] * zz / sum(zz)))

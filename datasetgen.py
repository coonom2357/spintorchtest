"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot
from vecenc import fsk_encode, qam_encode
from randvec import randvec

import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")
dsize = 300
fsklist = []
for i in range(dsize):
    vec = randvec(3, min_value=1, max_value=20)
    fsk_wave, fsk_t = fsk_encode(vec, samp_per_symbol=100, freq_min=1, freq_max=10)
    fsklist.append(fsk_wave)
fsk_dataset = np.array(fsklist)
"""Parameters"""
dx = 50e-9      # discretization (m)
dy = 50e-9      # discretization (m)
dz = 20e-9      # discretization (m)
nx = 100        # size x    (cells)
ny = 100        # size y    (cells)

Ms = 140e3      # saturation magnetization (A/m)
B0 = 60e-3      # bias field (T)
Bt = 1e-3       # excitation field amplitude (T)

dt = 20e-12     # timestep (s)
f1 = 4e9        # source frequency (Hz)
timesteps = 600 # number of timesteps for wave propagation


'''Directories'''
basedir = 'focus_Ms/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
## Here are three geometry modules initialized, just uncomment one of them to try:
Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
r0, dr, dm, z_off = 15, 4, 2, 10  # starting pos, period, magnet size, z distance
rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)
rho = torch.rand((rx, ry))*4 -2  # Design parameter array
geom = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0, 
                                    r0, dr, dm, z_off, rx, ry, Ms_CoPt)
# B1 = 50e-3      # training field multiplier (T)
# geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
# geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)
probes = []
Np = 3  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(Np+1)), 2))
model = spintorch.MMSolver(geom, dt, [src], probes)

dev = torch.device('cuda')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU


'''Define the source signal and output goal'''
t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
X = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude

INPUTS = X  # here we could cat multiple inputs
OUTPUTS = torch.tensor([int(Np/2)]).to(dev) # desired output


tic()
u = model(INPUTS).sum(dim=1)
stat_cuda('after forward')
stat_cuda('after backward')
toc()   

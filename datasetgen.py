"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot
from vecenc import fsk_encode, qam_encode
from randvec import randvec
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")

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

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
'''Generate FSK dataset'''

# dsize = 200  # number of samples to generate
# input_waves = []
# output_waves = []
# vectors = []

# for i in tqdm(range(dsize), desc="Generating dataset", unit="sample"):
#     model.retain_history = True
#     vec = randvec(3, min_value=1, max_value=20)
#     fsk_wave, fsk_t = fsk_encode(vec, samp_per_symbol=100, freq_min=1, freq_max=10)
#     INPUTS = Bt*torch.tensor(fsk_wave, device=dev).unsqueeze(0).unsqueeze(2)
#     output = model(INPUTS)
    
#     # Store the data
#     input_waves.append(INPUTS.squeeze())
#     output_waves.append(output.detach())
#     vectors.append(torch.tensor(vec))
    

# # Create dataset dictionary
# dataset = {
#     'input_waves': torch.stack(input_waves),      # Input FSK waves
#     'output_waves': torch.stack(output_waves),    # Model outputs
#     'vectors': torch.stack(vectors)               # Original vectors
# }

# # Save the dataset
# torch.save(dataset, savedir + 'fsk_dataset.pt')
# print(f"Dataset saved to {savedir}fsk_dataset.pt")
# print(f"Input shape: {dataset['input_waves'].shape}")
# print(f"Output shape: {dataset['output_waves'].shape}")
# print(dataset['output_waves'])
# torch.save(geom.rho, savedir + 'geometry_rho.pt')

# if model.retain_history:
#         with torch.no_grad():
#             spintorch.plot.geometry(model, epoch=dsize, plotdir=plotdir)
#             mz = torch.stack(model.m_history, 1)[0,:,2,]-model.m0[0,2,].unsqueeze(0).cpu()
#             wave_snapshot(model, mz[timesteps-1], (plotdir+f'snapshot_time{timesteps}_epoch{dsize}.png'),r"$m_z$")
#             wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+f'snapshot_time{int(timesteps/2)}_epoch{dsize}.png'),r"$m_z$")
#             wave_integrated(model, mz, (plotdir+f'integrated_epoch{dsize}.png'))


"""Optimize a focusing model"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot, geometry
from Encoding.vecenc import fsk_encode, qam_encode
from Encoding.randvec import randvec
from tqdm import tqdm
import warnings
import os
from model_parameter import model_parameters_init
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")

"""Parameters"""
dx = 50e-9      # discretization (m)
dy = 50e-9      # discretization (m)
dz = 20e-9      # discretization (m)
nx = 30        # size x    (cells)
ny = 30        # size y    (cells)
xmin = 7       # minimum x boundary for region of magnetization
ymin = 7       # minimum y boundary for region of magnetization
xmax = nx - xmin  # maximum x boundary for region of magnetization
ymax = ny - ymin  # maximum y boundary for region of magnetization
Ms = 140e3      # saturation magnetization (A/m)
B0 = 60e-3      # bias field (T)
Bt = 1e-3       # excitation field amplitude (T)
dt = 20e-12     # timestep (s)
f1 = 4e9        # source frequency (Hz)
timesteps = 300  # number of timesteps for wave propagation

'''Directories'''
basedir = 'contm_train/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
## Here are three geometry modules initialized, just uncomment one of them to try:
# Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
# r0, dr, dm, z_off = 10, 5, 2, 10  # starting pos, period, magnet size, z distance
# rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)
# rho = torch.rand((rx, ry))*4 -2  # Design parameter array
# geom = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0, 
#                                     r0, dr, dm, z_off, rx, ry, Ms_CoPt)
B1 = 50e-3      # training field multiplier (T)
geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms, random_init=True, init_scale=4.0, x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)
# geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)

'''Source setup'''
r0 = 5        # source line x-start
c0 = 0      # source line y-start
r1 = 5        # source line x-end
c1 = ny-1    # source line y-end
dim = 2        # source line dimension

'''Source setup - parameters'''
# All src parameters defined before creation
src_r0 = r0      # source line x-start
src_c0 = c0      # source line y-start
src_r1 = r1      # source line x-end
src_c1 = c1      # source line y-end
src_dim = dim    # source line dimension

src = spintorch.WaveLineSource(src_r0, src_c0, src_r1, src_c1, dim=src_dim)

'''Probes setup - parameters'''
# All probe parameters defined before creation
probe_Np = 2                    # number of probes
probe_x_factor = 0.94           # x-position as fraction of domain width (94% of nx)
probe_radius = 2                # probe disk radius (cells)
probe_x = int(nx * probe_x_factor)  # x-coordinate of all probes
probe_y_spacing = 1 / (probe_Np + 1)  # y-position spacing factor

probes = []
probe_params = []  # Store probe parameters for saving
for p in range(probe_Np):
    # y-position distributed evenly across domain: (p+1)/(Np+1)
    probe_y = int(ny * (p + 1) * probe_y_spacing)
    probes.append(spintorch.WaveIntensityProbeDisk(probe_x, probe_y, probe_radius))
    probe_params.append({"x": probe_x, "y": probe_y, "radius": probe_radius})

model = spintorch.MMSolver(geom, dt, [src], probes)
spintorch.plot.geometry(model, epoch=0, plotdir=plotdir)

model_parameters_init('contm_model', 'contm_train', dx, dy, dz, nx, ny, Ms, B0, Bt, dt, f1, timesteps, probe_params, src,
                      geometry_type='freeform', B1=B1,
                      random_init=True, init_scale=4.0, x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)

dev = torch.device('cuda')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

'''Generate dataset'''

dsize = 200  # number of samples to generate
input_waves_FSK = []
output_waves_FSK = []
input_waves_QAM = []
output_waves_QAM = []   
vectors = []

for i in tqdm(range(dsize), desc="Generating dataset", unit="sample"):
    model.retain_history = True
    vec = randvec(2, min_value=0, max_value=5)  # generate random vector
    #fsk encoding
    fsk_wave, fsk_t = fsk_encode(vec, samp_per_symbol=100, freq_min=f1, freq_max=f1*1.5, timesteps=timesteps)
    INPUTS_fsk = Bt*torch.tensor(fsk_wave, device=dev).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        output_fsk = model(INPUTS_fsk)

    #qam encoding
    qam_wave, qam_t = qam_encode(vec, samp_per_symbol=100, carrier_freq=f1, timesteps=timesteps)
    INPUTS_qam = Bt*torch.tensor(qam_wave, device=dev).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        output_qam = model(INPUTS_qam)

    # Store the data
    input_waves_FSK.append(INPUTS_fsk.squeeze())
    output_waves_FSK.append(output_fsk.detach())
    input_waves_QAM.append(INPUTS_qam.squeeze())
    output_waves_QAM.append(output_qam.detach())
    vectors.append(torch.tensor(vec))
    

# Create dataset dictionary
dataset = {
    'input_waves_FSK': torch.stack(input_waves_FSK),      # Input FSK waves
    'output_waves_FSK': torch.stack(output_waves_FSK),    # Model outputs FSK
    'input_waves_QAM': torch.stack(input_waves_QAM),      # Input QAM waves
    'output_waves_QAM': torch.stack(output_waves_QAM),    # Model outputs QAM
    'vectors': torch.stack(vectors)               # Original vectors
}

# Save the dataset
torch.save(dataset, savedir + 'dataset.pt')
torch.save(geom.rho, savedir + 'geometry_rho.pt')

# Save model state dict
torch.save(model.state_dict(), savedir + 'model_state_dict.pt')



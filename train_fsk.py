"""Train a focusing model on FSK dataset"""
import torch
import os

from tqdm import tqdm
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot
from torch.utils.data import Dataset, DataLoader

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
timesteps = 600 # number of timesteps for wave propagation

'''Directories'''
basedir = 'focus_Ms/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)

'''Load dataset'''
dataset_path = savedir + 'fsk_dataset.pt'
dataset_dict = torch.load(dataset_path)
print(f"Loaded dataset with {len(dataset_dict['input_waves'])} samples")

'''Create PyTorch Dataset'''
class FSKDataset(Dataset):
    def __init__(self, dataset_dict, device):
        # Input waves need to be shaped [batch, timesteps, sources]
        self.inputs = dataset_dict['input_waves'].unsqueeze(2).to(device)  # Add source dim at end
        self.outputs = dataset_dict['output_waves'].to(device)  # Target outputs from dataset
        self.vectors = dataset_dict['vectors'].to(device)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.vectors[idx]

'''Geometry, sources, probes, model definitions'''
Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
r0, dr, dm, z_off = 15, 4, 2, 10  # starting pos, period, magnet size, z distance
rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)

# Load geometry from dataset generation (or initialize new)
rho = torch.load(savedir + 'geometry_rho.pt')
# Don't use the loaded rho directly - let the geometry create its own parameter
# Or initialize fresh for training
rho_init = torch.rand((rx, ry)) * 4 - 2  # Random initialization for training
geom = spintorch.WaveGeometryArray(rho_init, (nx, ny), (dx, dy, dz), Ms, B0, 
                                    r0, dr, dm, z_off, rx, ry, Ms_CoPt)

src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)
probes = []
Np = 3  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-15, int(ny*(p+1)/(Np+1)), 2))
model = spintorch.MMSolver(geom, dt, [src], probes)

dev = torch.device('cuda')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)

# Verify parameters have gradients enabled
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

'''Create DataLoader'''
fsk_dataset = FSKDataset(dataset_dict, dev)
dataloader = DataLoader(fsk_dataset, batch_size=4, shuffle=True)

'''Define optimizer and loss function'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def mse_loss(predicted, target):
    """Mean squared error between predicted and target outputs"""
    return torch.nn.functional.mse_loss(predicted, target)

'''Load checkpoint'''
epoch_init = -1  # -1 = start from scratch
if epoch_init >= 0:
    checkpoint = torch.load(savedir + f'model_fsk_e{epoch_init}.pt')
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []

'''Train the network'''
num_epochs = 3
tic()

for epoch in tqdm(range(epoch_init+1, epoch_init+1+num_epochs), desc="Training epochs", unit="epoch"):
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, target_outputs, vectors) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Batches", unit="batch"):
        optimizer.zero_grad()
        
        # Process each sample in batch (model expects single samples)
        batch_predictions = []
        for i in range(inputs.size(0)):
            input_wave = inputs[i].unsqueeze(0)  # [1, timesteps, sources]
            u = model(input_wave)  # [1, timesteps, probes] - keep all timesteps
            batch_predictions.append(u)
        
        # Stack batch predictions
        batch_predictions = torch.cat(batch_predictions, dim=0)  # [batch, timesteps, probes]
        
        # Calculate loss against target outputs from dataset
        loss = mse_loss(batch_predictions, target_outputs.squeeze(1))
        epoch_loss += loss.item()
        num_batches += 1
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    # Average epoch loss
    avg_loss = epoch_loss / num_batches
    loss_iter.append(avg_loss)
    
    # Plot loss
    spintorch.plot.plot_loss(loss_iter, plotdir)
    
    print(f"Epoch {epoch} finished -- Avg Loss: {avg_loss:.6f}")
    toc()
    
    '''Save model checkpoint'''
    if epoch % 5 == 0:  # Save every 5 epochs
        torch.save({
            'epoch': epoch,
            'loss_iter': loss_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, savedir + f'model_fsk_e{epoch}.pt')
        print(f"Checkpoint saved at epoch {epoch}")

'''Plot final wave propagation with first sample'''
model.retain_history = True
with torch.no_grad():
    test_input = fsk_dataset.inputs[0].unsqueeze(0)
    u = model(test_input).sum(dim=1)
    target_output = fsk_dataset.outputs[0]
    
    # Plot predicted vs target
    spintorch.plot.plot_output(u[0,], 2, epoch, plotdir)  # Predicted output
    spintorch.plot.geometry(model, epoch=epoch, plotdir=plotdir)
    
    # Stack history and move to CPU for plotting
    mz = torch.stack(model.m_history, 1).cpu()[0,:,2,] - model.m0[0,2,].unsqueeze(0).cpu()
    wave_snapshot(model, mz[timesteps-1], (plotdir+f'snapshot_time{timesteps}_final.png'), r"$m_z$")
    wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+f'snapshot_time{int(timesteps/2)}_final.png'), r"$m_z$")
    wave_integrated(model, mz, (plotdir+f'integrated_final.png'))

print("Training complete!")

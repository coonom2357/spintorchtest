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
r0 = 5        # source line x-start
c0 = 0      # source line y-start
r1 = 5        # source line x-end
c1 = ny-1    # source line y-end
dim = 2        # source line dimension
dt = 20e-12     # timestep (s)
f1 = 4e9        # source frequency (Hz)

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
    
'''Directories'''
basedir = 'contm_train'
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
# print (rx, ry)
# rho = torch.rand((rx, ry))*4 -2  # Design parameter array
# geom = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0, 
#                                     r0, dr, dm, z_off, rx, ry, Ms_CoPt)
B1 = 50e-3      # training field multiplier (T)
geom = spintorch.WaveGeometryFreeForm((nx, ny), (dx, dy, dz), B0, B1, Ms)
# geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)
src = spintorch.WaveLineSource(r0, c0, r1, c1, dim=dim)
probes = []
Np = 2  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(int(nx*.94), int(ny*(p+1)/(Np+1)), 2))
model = spintorch.MMSolver(geom, dt, [src], probes)
spintorch.plot.geometry(model, epoch=0, plotdir=plotdir)
dev = torch.device('cuda')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU

# Verify parameters have gradients enabled
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")

'''Create DataLoader'''
fsk_dataset = FSKDataset(torch.load('2x3train/fsk_dataset.pt'), dev)
dataloader = DataLoader(fsk_dataset, batch_size=4, shuffle=True)

'''Define optimizer, scheduler, and loss function'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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

'''Calculate Timesteps'''
timesteps = fsk_dataset.inputs.shape[1]

'''Train the network'''
num_epochs = 20
tic()

for epoch in tqdm(range(epoch_init+1, epoch_init+1+num_epochs), desc="Training", unit="epoch"):
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, target_outputs, vectors) in enumerate(dataloader):
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
    scheduler.step(avg_loss) # Update learning rate based on loss

    
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
    mz = torch.stack(model.m_history, dim=0).squeeze(1)[:, 2, :, :].cpu() - model.m0[0, 2, :, :].unsqueeze(0).cpu()
    wave_snapshot(model, mz[timesteps-1], (plotdir+f'snapshot_time{timesteps}_final.png'), r"$m_z$")
    wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+f'snapshot_time{int(timesteps/2)}_final.png'), r"$m_z$")
    wave_integrated(model, mz, (plotdir+f'integrated_final.png'))

print("Training complete!")

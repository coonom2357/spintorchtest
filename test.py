import torch
import os
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot
from vecenc import fsk_encode, qam_encode
from randvec import randvec

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt

dataset = torch.load('models/focus_Ms/fsk_dataset.pt')

# Plot first input wave
plt.figure(figsize=(10, 5))
plt.plot(dataset['input_waves'][0].numpy(), alpha=0.7)
plt.title('First Input FSK Wave')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Plot first output (all 3 probes)
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.plot(dataset['output_waves'][0, 0, :, i].numpy(), label=f'Probe {i+1}', alpha=0.7)
plt.title('First Output Wave (3 Probes)')
plt.xlabel('Time step')
plt.ylabel('Intensity')
plt.legend()
plt.tight_layout()
plt.show()
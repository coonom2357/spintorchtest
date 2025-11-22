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
print('FSK dataset shape:', fsk_dataset.shape)
print('Sample FSK wave:', fsk_dataset[0])
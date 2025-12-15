from model_parameter import load_model_from_checkpoint, model_parameters_init
from display_wave import plot_input_output_model
import torch
import spintorch
import os

model, params = load_model_from_checkpoint('2x3train/model_fsk_e10.pt', '2x3train/2x3train_parameters.json')
plot_input_output_model(model, '2x3train/fsk_dataset.pt', 'plots/2x3train/', 2)
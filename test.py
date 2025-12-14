from model_parameter import load_model_from_checkpoint, model_parameters_init
import torch
import spintorch
import os

load_model_from_checkpoint('2x3train', '2x3train/model_fsk_e15.pt')

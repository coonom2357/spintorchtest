from model_parameter import load_model_from_checkpoint, model_parameters_init
from display_wave import plot_input_output_model, plot_multiple_models
from display_geometry import plot_geometry_model
import torch
import spintorch 
import os
from Encoding.vecenc import visualize_vector_encoding
vector=[3, 7, 2, 15]
visualize_vector_encoding(vector=vector, plot_dir="./plots", timesteps=len(vector)*300)
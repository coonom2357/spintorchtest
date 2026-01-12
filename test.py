from model_parameter import load_model_from_checkpoint, model_parameters_init
from display_wave import plot_input_output_model, plot_multiple_models
import torch
import spintorch
import os

# Plot all models in 2x3train_w_scheduler
model_dir = "models/2x3train_w_scheduler/"
dataset_path = "2x3train/fsk_dataset.pt"
parameter_path = "2x3train/2x3train_parameters.json"
plot_dir = "plots/2x3train_w_scheduler/"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Create list of models with their names
model_list = [
    (f"{model_dir}model_fsk_e0.pt", "Epoch 0"),
    (f"{model_dir}model_fsk_e5.pt", "Epoch 5"),
    (f"{model_dir}model_fsk_e10.pt", "Epoch 10"),
    (f"{model_dir}model_fsk_e15.pt", "Epoch 15"),
]

print("Plotting all models from 2x3train_w_scheduler...")
plot_multiple_models(model_list, dataset_path, plot_dir, index=3, name="2x3train_w_scheduler", parameter_path=parameter_path)
print(f"Multi-model plot saved to {plot_dir}")

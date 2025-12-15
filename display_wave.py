import torch
import spintorch
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_input_output_model(model, dataset_path, plot_dir, index, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Visualize input and output waves of the model and compare with desired output.
    
    Args:
        model_path (str): Path to the trained model file
        dataset_path (str): Path to the dataset file
        plot_dir (str): Directory to save the plots
        index (int): Index of the data sample to visualize
        device (str): Device to run on ('cuda' or 'cpu')
    """
    print(f"Running on device: {device}")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    dataset = torch.load(dataset_path, map_location=device)
    wave_input = dataset['input_waves'][index].to(device)
    wave_desired = dataset['output_waves'][index].to(device)
    
    # Prepare input for model (add batch and channel dims)
    input_tensor = wave_input.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, timesteps)
    
    with torch.no_grad():  # Disable gradients for inference
        wave_output = model(input_tensor)
    
    # For plotting, squeeze extra dims and move to CPU
    wave_output = wave_output.squeeze(0).cpu()  # Shape: (Np, timesteps) where Np is number of probes
    wave_input = wave_input.cpu()
    wave_desired = wave_desired.cpu()
    
    t = np.linspace(0, len(wave_input), len(wave_input))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot input wave
    ax1.plot(t, wave_input.numpy(), color='green', linewidth=0.8)
    ax1.set_title('Input Wave')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot desired output wave (first probe)
    ax2.plot(t, wave_desired[0, 0].numpy(), color='blue', linewidth=0.8)
    ax2.set_title('Desired Output Wave (Probe 1)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot model output wave (first probe)
    ax3.plot(t, wave_output[0].numpy(), color='red', linewidth=0.8)
    ax3.set_title('Model Output Wave (Probe 1)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'wave_comparison_index_{index}.png'))
    plt.show()

if __name__ == "__main__":
    model_path = "2x3train/model_fsk_e10.pt"
    dataset_path = "2x3train/fsk_dataset.pt"
    plot_dir = "plots/2x3train/"
    index = 0  # Index of the sample to visualize

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    print("Plotting input and output waves...")
    plot_input_output_model(model_path, dataset_path, plot_dir, index)
    print(f"Plots saved to {plot_dir}")
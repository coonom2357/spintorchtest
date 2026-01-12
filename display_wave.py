import torch
import spintorch
import matplotlib.pyplot as plt
import os
import numpy as np
from model_parameter import load_model_from_checkpoint


def plot_input_output_model(model, dataset_path, plot_dir, index, epoch, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Visualize input and output waves of the model and compare with desired output.
    
    Args:
        model (torch.nn.Module or str): Trained model object or path to model file
        dataset_path (str): Path to the dataset file
        plot_dir (str): Directory to save the plots
        index (int): Index of the data sample to visualize
        epoch (int): Current epoch number for naming the plot file
        model_name (str): Name/identifier for the model (for plot title). If None, uses model.__class__.__name__
        device (str): Device to run on ('cuda' or 'cpu')
    """
    print(f"Running on device: {device}")
    
    # Load model if path is provided
    if isinstance(model, str):
        print(f"Loading model from {model}...")
        model = torch.load(model, map_location=device)
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    dataset = torch.load(dataset_path, map_location=device)
    wave_input = dataset['input_waves'][index].to(device)
    wave_desired = dataset['output_waves'][index].to(device)
    
    # Prepare input for model (add source dim)
    input_tensor = wave_input.unsqueeze(0).unsqueeze(-1)  # (timesteps,) -> (1, timesteps) -> (1, timesteps, 1)
    
    print(f"Input shape: {wave_input.shape}")
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Desired output shape: {wave_desired.shape}")
    
    with torch.no_grad():  # Disable gradients for inference
        wave_output = model(input_tensor)
    print(f"Output tensor shape: {wave_output.shape}")

    # For plotting, squeeze extra dims and move to CPU
    wave_output = wave_output.squeeze(0).cpu()  # Shape: (timesteps, Np) where Np is number of probes
    wave_input = wave_input.cpu()
    wave_desired = wave_desired.squeeze(0).cpu()  # Remove batch dimension to match wave_output
    
    # Ensure wave_output is 2D: (timesteps, Np)
    if wave_output.dim() == 1:
        wave_output = wave_output.unsqueeze(1)
    
    t = np.linspace(0, len(wave_input), len(wave_input))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot input wave
    ax1.plot(t, wave_input.numpy(), color='green', linewidth=0.8)
    ax1.set_title('Input Wave')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot desired output and model output overlapping (first probe)
    ax2.plot(t, wave_desired[:, 0].numpy(), color='blue', linewidth=1.2, label='Desired Output', alpha=0.8)
    ax2.plot(t, wave_output[:, 0].numpy(), color='red', linewidth=1.0, label='Model Output', alpha=0.8, linestyle='--')
    ax2.set_title(f'Output Comparison - {epoch} (Probe 1)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'wave_comparison_index_I{index}_epoch_E{epoch}.png'))
    plt.show()


def plot_multiple_models(model_list=None, dataset_path=None, plot_dir=None, index=None, name=None, parameter_path=None, device='cpu'):
    """
    Plot and compare multiple models on the same graph.
    
    Args:
        model_list (list): List of tuples (model_path_or_object, model_name)
        dataset_path (str): Path to the dataset file
        plot_dir (str): Directory to save the plots
        index (int): Index of the data sample to visualize
        name (str): Name for the comparison plot file
        parameter_path (str): Path to the parameters JSON file (required if model_list contains paths)
        device (str): Device to run on ('cuda' or 'cpu')
    """
    print(f"Running on device: {device}")
    
    dataset = torch.load(dataset_path, map_location=device)
    wave_input = dataset['input_waves'][index]
    vector_input = dataset['vectors'][index]

    # If no models provided, just plot input wave
    if model_list is None:
        t = np.linspace(0, len(wave_input), len(wave_input))
        fig, ax = plt.subplots(figsize=(12, 5))
        vector_text = f"Vector: {vector_input.numpy().tolist()}"
        ax.plot(t, wave_input.numpy(), color='green', linewidth=0.8)
        ax.set_title('Input Wave')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, vector_text, transform=ax.transAxes, fontsize=10)

        plt.tight_layout()
        plot_name = name or f"input_wave_index_I{index}.png"
        plt.savefig(os.path.join(plot_dir, plot_name))
        plt.show()
        return

    wave_desired = dataset['output_waves'][index].squeeze(0)
    
    t = np.linspace(0, len(wave_input), len(wave_input))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot input wave
    ax1.plot(t, wave_input.numpy(), color='green', linewidth=0.8)
    ax1.set_title('Input Wave')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f"Vector: {vector_input.numpy().tolist()}", transform=ax1.transAxes, fontsize=10)
    
    # Plot desired output
    ax2.plot(t, wave_desired[:, 0].numpy(), color='blue', linewidth=1.2, label='Desired Output', alpha=0.8)
    
    # Plot each model's output
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    linestyles = ['--', '-.', ':', '-', '--', '-.', ':', '-']
    
    for idx, (model, model_name) in enumerate(model_list):
        if isinstance(model, str):
            print(f"Loading model {model_name} from {model}...")
            if parameter_path is None:
                raise ValueError("parameter_path is required when loading models from file paths")
            model, _ = load_model_from_checkpoint(model, parameter_path)
        
        model.to(device)
        model.eval()
        
        # Prepare input for model
        input_tensor = wave_input.unsqueeze(0).unsqueeze(-1).to(device)
        
        with torch.no_grad():
            wave_output = model(input_tensor)
        
        wave_output = wave_output.squeeze(0).cpu()
        if wave_output.dim() == 1:
            wave_output = wave_output.unsqueeze(1)
        
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        ax2.plot(t, wave_output[:, 0].numpy(), color=color, linewidth=1.0, 
                label=f'{model_name}', alpha=0.8, linestyle=linestyle)
    
    ax2.set_title(f'Output Comparison - Multiple Models (Probe 1)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'multi_model_wave_comparison_{name}_index_I{index}.png'))
    plt.show()

if __name__ == "__main__":
    # Single model example
    model_path = "2x3train/model_fsk_e10.pt"
    dataset_path = "2x3train/fsk_dataset.pt"
    plot_dir = "plots/2x3train/"
    index = 0  # Index of the sample to visualize

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot single model
    print("Plotting single model...")
    plot_input_output_model(model_path, dataset_path, plot_dir, index, epoch=10,)
    print(f"Plot saved to {plot_dir}")
    
    # Multiple models example (uncomment to use)
    # model_list = [
    #     ("2x3train/model_fsk_e0.pt", "Epoch 0"),
    #     ("2x3train/model_fsk_e5.pt", "Epoch 5"),
    #     ("2x3train/model_fsk_e10.pt", "Epoch 10"),
    #     ("2x3train/model_fsk_e15.pt", "Epoch 15"),
    # ]
    # print("Plotting multiple models...")
    # plot_multiple_models(model_list, dataset_path, plot_dir, index, epoch="all")
    # print(f"Multi-model plot saved to {plot_dir}")
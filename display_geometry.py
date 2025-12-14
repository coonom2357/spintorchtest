import torch
import spintorch
import matplotlib.pyplot as plt
import os
import numpy as np

# Parameters (must match the ones used when geometry was saved)
dx = 50e-9      # discretization (m)
dy = 50e-9      # discretization (m)
dz = 20e-9      # discretization (m)
nx = 30        # size x (cells)
ny = 30        # size y (cells)
Ms = 140e3      # saturation magnetization (A/m)
B0 = 60e-3      # bias field (T)
Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
r0, dr, dm, z_off = 10, 5, 2, 10  # starting pos, period, magnet size, z distance
rx, ry = int((nx-2*r0)/dr), int((ny-2*r0)/dr+1)
dt = 20e-12     # timestep (s)

def display_geometry_from_file(geometry_path):
    """
    Load and display geometry from a saved geometry_rho.pt file

    Args:
        geometry_path (str): Path to the geometry_rho.pt file
    """
    try:
        # Load the saved rho parameter
        rho = torch.load(geometry_path, map_location='cpu')
        print(f"Loaded geometry rho with shape: {rho.shape}")
        print(f"Expected shape: ({rx}, {ry}) = {(rx, ry)}")

        # Create geometry using the loaded rho
        geom = spintorch.WaveGeometryArray(rho, (nx, ny), (dx, dy, dz), Ms, B0,
                                            r0, dr, dm, z_off, rx, ry, Ms_CoPt)

        # Create source and probes (same as in training)
        src = spintorch.WaveLineSource(5, 0, 5, ny-1, dim=2)
        probes = []
        Np = 2  # number of probes
        for p in range(Np):
            probes.append(spintorch.WaveIntensityProbeDisk(int(nx*.94), int(ny*(p+1)/(Np+1)), 2))

        # Create model
        model = spintorch.MMSolver(geom, dt, [src], probes)

        # Create output directory
        plot_dir = "2x2train/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Display and save the geometry
        print("Displaying and saving geometry...")
        spintorch.plot.geometry(model, outline=False, plotdir=plot_dir, epoch=0)
        print(f"Geometry plot saved to {plot_dir}/geometry_epoch0.png")

        # Also show the rho values
        print("\nGeometry rho values:")
        print(f"Shape: {rho.shape}")
        print(f"Min: {rho.min().item():.3f}")
        print(f"Max: {rho.max().item():.3f}")
        print(f"Mean: {rho.mean().item():.3f}")

    except FileNotFoundError:
        print(f"Error: File '{geometry_path}' not found.")
        print("Make sure the geometry_rho.pt file exists in the specified location.")
    except Exception as e:
        print(f"Error loading/displaying geometry: {e}")

def plot_geometry_model(model_path, plot_dir="2x2train/"):
    """
    Plot the geometry of the model and save the figure.

    Args:
        model (spintorch.MMSolver): The MMSolver model containing the geometry.
        plot_dir (str): Directory to save the plot.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    model = torch.load(model_path, map_location='cpu')
    spintorch.plot.geometry(model, outline=False, plotdir=plot_dir, epoch=0)
    print(f"Geometry plot saved to {plot_dir}/geometry_epoch0.png")

if __name__ == "__main__":
    # Try the 2x3train directory first since that's where the files are
    geometry_path = "2x3train/geometry_rho.pt"
    plot_dir = "2x3train/"

    print("Geometry Display Tool")
    print("====================")
    print(f"Loading geometry from: {geometry_path}")
    print(f"Saving plots to: {plot_dir}/")

    if os.path.exists(geometry_path):
        display_geometry_from_file(geometry_path)
        print(f"\nPlots saved successfully to {plot_dir}/ directory")
    else:
        print(f"File not found at: {geometry_path}")
        print("Please check that the file exists and try again.")
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

def plot_geometry_model(model_path, plot_dir):
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

def display_geometry_from_checkpoint(checkpoint_path, parameter_path, plot_dir, epoch=0, geometry_type=None):
    """
    Load geometry from a model checkpoint's state_dict and display it.
    Supports all spintorch geometry types: WaveGeometryArray, WaveGeometryFreeForm, WaveGeometryMs
    
    Args:
        checkpoint_path (str): Path to the checkpoint file (.pt)
        parameter_path (str): Path to the parameters JSON file
        plot_dir (str): Directory to save the plot
        epoch (int): Epoch number for the output filename
        geometry_type (str): Type of geometry ('array', 'freeform', 'ms', or None for auto-detect)
    """
    import json
    
    try:
        # Load parameters from JSON
        with open(parameter_path, 'r') as f:
            params = json.load(f)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Auto-detect geometry type if not specified
        if geometry_type is None:
            if 'geom.rho' in state_dict or any('rho' in key.lower() for key in state_dict.keys()):
                geometry_type = 'array'
            elif 'geom.B1' in state_dict or any('B1' in key for key in state_dict.keys()):
                geometry_type = 'freeform'
            else:
                geometry_type = 'ms'  # Default to Ms geometry
            print(f"Auto-detected geometry type: {geometry_type}")
        
        # Create output directory
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Create source and probes (same for all geometry types)
        src_p = params['src_params']
        src = spintorch.WaveLineSource(src_p[0], src_p[1], src_p[2], src_p[3], dim=2)
        
        probes = []
        for p in range(params['Np']):
            probes.append(spintorch.WaveIntensityProbeDisk(
                params['probe_x'],
                int(params['ny'] * (p+1) / (params['Np']+1)),
                2
            ))
        
        # Create geometry based on detected type
        if geometry_type == 'array':
            print("Loading WaveGeometryArray...")
            # Extract rho from state_dict
            rho = None
            for key, value in state_dict.items():
                if 'rho' in key.lower():
                    rho = value
                    print(f"Found rho in state_dict key: {key}")
                    break
            
            if rho is None:
                print("Error: Could not find 'rho' parameter in checkpoint state_dict")
                print(f"Available keys: {list(state_dict.keys())}")
                return
            
            print(f"Geometry rho shape: {rho.shape}")
            
            geom = spintorch.WaveGeometryArray(
                rho, 
                (params['nx'], params['ny']),
                (params['dx'], params['dy'], params['dz']),
                params['Ms'], params['B0'],
                params['r0'], params['dr'], params['dm'], params['z_off'],
                params['rx'], params['ry'], params['Ms_CoPt']
            )
            
            print("\nGeometry rho values:")
            print(f"Shape: {rho.shape}")
            print(f"Min: {rho.min().item():.3f}")
            print(f"Max: {rho.max().item():.3f}")
            print(f"Mean: {rho.mean().item():.3f}")
            print(f"Std: {rho.std().item():.3f}")
        
        elif geometry_type == 'freeform':
            print("Loading WaveGeometryFreeForm...")
            # Check if we have B1 in parameters
            B1 = params.get('B1', 50e-3)  # Default value
            
            geom = spintorch.WaveGeometryFreeForm(
                (params['nx'], params['ny']),
                (params['dx'], params['dy'], params['dz']),
                params['B0'], B1, params['Ms']
            )
            print(f"Geometry: FreeForm with B1={B1:.3e} T")
        
        elif geometry_type == 'ms':
            print("Loading WaveGeometryMs...")
            geom = spintorch.WaveGeometryMs(
                (params['nx'], params['ny']),
                (params['dx'], params['dy'], params['dz']),
                params['Ms'], params['B0']
            )
            print(f"Geometry: Ms with saturation magnetization Ms={params['Ms']:.3e} A/m")
        
        else:
            print(f"Error: Unknown geometry type '{geometry_type}'")
            print("Supported types: 'array', 'freeform', 'ms'")
            return
        
        # Create model
        model = spintorch.MMSolver(geom, params['dt'], [src], probes)
        
        # Display and save the geometry
        print("Displaying and saving geometry...")
        spintorch.plot.geometry(model, outline=False, plotdir=plot_dir, epoch=epoch)
        print(f"✓ Geometry plot saved to {plot_dir}/geometry_epoch{epoch}.png")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error loading/displaying geometry from checkpoint: {e}")
        import traceback
        traceback.print_exc()

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
    
    # Examples: Load geometry from checkpoint state_dict
    # Uncomment and modify paths to use:
    
    # WaveGeometryArray (auto-detect):
    # display_geometry_from_checkpoint(
    #     checkpoint_path="2x3train/model_fsk_e10.pt",
    #     parameter_path="2x3train/2x3train_parameters.json",
    #     plot_dir="2x3train/checkpoint_geometry/",
    #     epoch=10
    # )
    
    # Explicit geometry type specification:
    # display_geometry_from_checkpoint(
    #     checkpoint_path="2x3train/model_fsk_e10.pt",
    #     parameter_path="2x3train/2x3train_parameters.json",
    #     plot_dir="2x3train/checkpoint_geometry/",
    #     epoch=10,
    #     geometry_type='array'  # 'array', 'freeform', or 'ms'
    # )
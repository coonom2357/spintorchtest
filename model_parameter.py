import os
import torch
import spintorch
import json
def model_parameters_init(modelname, dx, dy, dz, nx, ny, Ms, B0, Bt, dt, f1, timesteps, Ms_CoPt, Np, r0=10, dr=5, dm=2, z_off=10):
    """
    Initialize and save model parameters to JSON file.
    
    Args:
        modelname (str): Name of the model (used for directory and file naming)
        dx (float): Discretization step in x-direction (m)
        dy (float): Discretization step in y-direction (m)
        dz (float): Discretization step in z-direction (m)
        nx (int): Number of cells in x-direction
        ny (int): Number of cells in y-direction
        Ms (float): Saturation magnetization (A/m)
        B0 (float): Bias magnetic field (T)
        Bt (float): Excitation field amplitude (T)
        dt (float): Timestep for simulation (s)
        f1 (float): Source frequency (Hz)
        timesteps (int): Number of timesteps for wave propagation
        Ms_CoPt (float): Saturation magnetization of nanomagnets (A/m)
        Np (int): Number of probes
        r0 (int): Starting position for magnet array (default: 10)
        dr (int): Period/spacing of magnet array (default: 5)
        dm (int): Magnet size (default: 2)
        z_off (int): Z-distance offset (default: 10)
    """
    # Calculate geometry dimensions
    rx = int((nx - 2*r0) / dr)
    ry = int((ny - 2*r0) / dr + 1)
    
    src = spintorch.WaveLineSource(5, 0, 5, ny-1, dim=2)
    probes = []
    for p in range(Np):
        probes.append(spintorch.WaveIntensityProbeDisk(int(nx*.94), int(ny*(p+1)/(Np+1)), 2))
    
    data = { 
        "dx": dx, "dy": dy, "dz": dz, "nx": nx, "ny": ny, "Ms": Ms, "B0": B0,
        "Bt": Bt, "dt": dt, "f1": f1, "timesteps": timesteps, "Ms_CoPt": Ms_CoPt,
        "r0": r0, "dr": dr, "dm": dm, "z_off": z_off, "rx": rx, "ry": ry, "Np": Np,
        "src_params": [5, 0, 5, ny-1],
        "probe_x": int(nx*.94), "probe_y_offset": 1/(Np+1)
    }
    with open(f'{modelname}/{modelname}_parameters.json', 'w') as f:
        json.dump(data, f)
    
def load_model_from_checkpoint(modelname, checkpoint_path):
    """
    Reconstruct model from saved parameters and checkpoint.
    
    Args:
        modelname (str): Name of the model (matches the directory/file name used in model_parameters_init)
        checkpoint_path (str): Path to the checkpoint file (.pt)
    
    Returns:
        tuple: (model, params) where model is the reconstructed MMSolver and params is the loaded parameters dict
    """
    # Load parameters from JSON
    with open(f'{modelname}_parameters.json', 'r') as f:
        params = json.load(f)
    
    # Create geometry with the SAME rx, ry (not randomized)
    rho = torch.rand((params['rx'], params['ry'])) * 4 - 2  # OK to randomize rho
    geom = spintorch.WaveGeometryArray(
        rho, 
        (params['nx'], params['ny']),
        (params['dx'], params['dy'], params['dz']),
        params['Ms'], params['B0'],
        params['r0'], params['dr'], params['dm'], params['z_off'],
        params['rx'], params['ry'], params['Ms_CoPt']
    )
    
    # Recreate src with saved parameters
    src_p = params['src_params']
    src = spintorch.WaveLineSource(src_p[0], src_p[1], src_p[2], src_p[3], dim=2)
    
    # Recreate probes with saved parameters
    probes = []
    for p in range(params['Np']):
        probes.append(spintorch.WaveIntensityProbeDisk(
            params['probe_x'], 
            int(params['ny'] * (p+1) / (params['Np']+1)), 
            2
        ))
    
    # Create model
    model = spintorch.MMSolver(geom, params['dt'], [src], probes)
    
    # Load trained weights from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, params


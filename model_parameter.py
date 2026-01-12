import os
import torch
import spintorch
import json
def model_parameters_init(modelname, basedir, dx, dy, dz, nx, ny, Ms, B0, Bt, dt, f1, timesteps, probe_params, src,
                          geometry_type='array', B1=None, Ms_CoPt=None, 
                          array_r0=10, array_dr=5, array_dm=2, array_z_off=10,
                          random_init=False, init_scale=4.0, x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Initialize and save model parameters to JSON file.
    
    Args:
        modelname (str): Name of the model (used for directory and file naming)
        basedir (str): Base directory for saving parameters
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
        probe_params (list): List of probe parameter dicts with keys 'x', 'y', 'radius'
        src (WaveLineSource): Source object to save parameters from
        geometry_type (str): Type of geometry - 'array', 'freeform', or 'ms' (default: 'array')
        B1 (float): Excitation field strength (required for 'freeform', default: None)
        Ms_CoPt (float): Saturation magnetization of nanomagnets (required for 'array', default: None)
        array_r0 (int): Starting position for magnet array (default: 10, only for 'array')
        array_dr (int): Period/spacing of magnet array (default: 5, only for 'array')
        array_dm (int): Magnet size (default: 2, only for 'array')
        array_z_off (int): Z-distance offset (default: 10, only for 'array')
        random_init (bool): Whether to randomly initialize (for 'freeform', default: False)
        init_scale (float): Scale of random initialization (default: 4.0, only for 'freeform')
        x_min, x_max, y_min, y_max (int): Region bounds for random initialization (only for 'freeform')
    """
    Np = len(probe_params)  # Number of probes
    
    # Base parameters common to all geometry types (including src and probes)
    data = { 
        "geometry_type": geometry_type,
        "dx": dx, "dy": dy, "dz": dz, "nx": nx, "ny": ny, "Ms": Ms, "B0": B0,
        "Bt": Bt, "dt": dt, "f1": f1, "timesteps": timesteps, "Np": Np,
        # Source parameters - saved for all geometry types
        "src_r0": int(src.r0), "src_c0": int(src.c0), "src_r1": int(src.r1), "src_c1": int(src.c1), 
        "src_dim": int(src.dim),
        # Probe parameters - save each probe's position and radius
        "probes": probe_params
    }
    
    # Add geometry-specific parameters
    if geometry_type == 'array':
        if Ms_CoPt is None:
            raise ValueError("Ms_CoPt is required for 'array' geometry type")
        rx = int((nx - 2*array_r0) / array_dr)
        ry = int((ny - 2*array_r0) / array_dr + 1)
        data.update({
            "Ms_CoPt": Ms_CoPt,
            "array_r0": array_r0, "array_dr": array_dr, "array_dm": array_dm, "array_z_off": array_z_off, 
            "rx": rx, "ry": ry
        })
    elif geometry_type == 'freeform':
        if B1 is None:
            raise ValueError("B1 is required for 'freeform' geometry type")
        data["B1"] = B1
        # Save random initialization parameters
        data["random_init"] = random_init
        data["init_scale"] = init_scale
        if x_min is not None:
            data["x_min"] = x_min
        if x_max is not None:
            data["x_max"] = x_max
        if y_min is not None:
            data["y_min"] = y_min
        if y_max is not None:
            data["y_max"] = y_max
    elif geometry_type == 'ms':
        # WaveGeometryMs only needs Ms and B0, which are already in data
        pass
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}. Must be 'array', 'freeform', or 'ms'")
    
    with open(f'{basedir}/{modelname}_parameters.json', 'w') as f:
        json.dump(data, f)

def load_model_from_checkpoint(checkpoint_path, parameter_path):
    """
    Reconstruct model from saved parameters and checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file (.pt)
        parameter_path (str): Path to the parameters JSON file
    
    Returns:
        tuple: (model, params) where model is the reconstructed MMSolver and params is the loaded parameters dict
    """
    # Load parameters from JSON
    with open(parameter_path, 'r') as f:
        params = json.load(f)
    
    geometry_type = params.get('geometry_type', 'array')  # Default to 'array' for backward compatibility
    
    # Recreate src with saved parameters
    src = spintorch.WaveLineSource(
        int(params['src_r0']), int(params['src_c0']), 
        int(params['src_r1']), int(params['src_c1']), 
        dim=int(params.get('src_dim', 2))
    )
    
    # Recreate probes with saved parameters
    if 'probes' in params:
        # New format: saved probe parameters
        probes = []
        for probe_params in params['probes']:
            probes.append(spintorch.WaveIntensityProbeDisk(
                probe_params['x'],
                probe_params['y'],
                probe_params['radius']
            ))
    else:
        # Backward compatibility: reconstruct from old probe_x and probe_y_offset
        probes = []
        for p in range(params['Np']):
            probes.append(spintorch.WaveIntensityProbeDisk(
                params['probe_x'], 
                int(params['ny'] * (p+1) / (params['Np']+1)), 
                2
            ))
    
    # Create geometry based on geometry_type
    if geometry_type == 'array':
        rho = torch.rand((params['rx'], params['ry'])) * 4 - 2  # OK to randomize rho
        geom = spintorch.WaveGeometryArray(
            rho, 
            (params['nx'], params['ny']),
            (params['dx'], params['dy'], params['dz']),
            params['Ms'], params['B0'],
            params['array_r0'], params['array_dr'], params['array_dm'], params['array_z_off'],
            params['rx'], params['ry'], params['Ms_CoPt']
        )
    elif geometry_type == 'freeform':
        B1 = params.get('B1', 50e-3)  # Default value if not specified
        random_init = params.get('random_init', False)
        init_scale = params.get('init_scale', 4.0)
        x_min = params.get('x_min', None)
        x_max = params.get('x_max', None)
        y_min = params.get('y_min', None)
        y_max = params.get('y_max', None)
        
        geom = spintorch.WaveGeometryFreeForm(
            (params['nx'], params['ny']),
            (params['dx'], params['dy'], params['dz']),
            params['B0'], B1, params['Ms'],
            random_init=random_init, init_scale=init_scale,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
    elif geometry_type == 'ms':
        geom = spintorch.WaveGeometryMs(
            (params['nx'], params['ny']),
            (params['dx'], params['dy'], params['dz']),
            params['Ms'], params['B0']
        )
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}. Must be 'array', 'freeform', or 'ms'")
    
    # Create model
    model = spintorch.MMSolver(geom, params['dt'], [src], probes)
    
    # Use map_location to handle GPU->CPU loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, params


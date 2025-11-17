import numpy as np
import matplotlib.pyplot as plt


def FSK(matrix, normalize=True, samp_per_ele=100):
    """
    Convert a matrix into a sinusoidal wave representation.
    
    Args:
        matrix (list or np.array): Input matrix
        normalize (bool): Whether to normalize values to [0, 2π] range
    
    Returns:
        np.array: 1D array of sinusoidal wave values
        np.array: Time/position array
    """
    # Convert to numpy array and flatten
    matrix = np.array(matrix)
    flat_values = matrix.flatten()
    
    # Normalize to [0, 2π] range if requested
    if normalize:
        min_val = flat_values.min()
        max_val = flat_values.max()
        if max_val > min_val:
            normalized = (flat_values - min_val) / (max_val - min_val) * 2 * np.pi
    else:
        normalized = flat_values
    
    # Generate sinusoidal wave
    t = np.linspace(0, len(normalized), len(normalized) * samp_per_ele)
    wave = np.zeros(len(t))
    
    # Create wave by summing sinusoids at each matrix position
    for i, phase in enumerate(normalized):
        frequency = (i + 1) / len(normalized)  # Varying frequency
        wave += np.sin(2 * np.pi * frequency * t + phase)
    
    return wave, t


def AM(matrix, samp_per_ele=100):
    """
    Convert matrix values to amplitude-modulated sinusoidal wave.
    Each matrix element becomes an amplitude at a specific time point.
    
    Args:
        matrix (list or np.array): Input matrix
    
    Returns:
        np.array: Wave signal
        np.array: Time array
    """
    matrix = np.array(matrix)
    flat_values = matrix.flatten()
    
    # Generate time points
    t = np.linspace(0, len(flat_values), len(flat_values) * samp_per_ele)
    
    # Base frequency
    base_freq = 1.0
    carrier = np.sin(2 * np.pi * base_freq * t)
    
    # Amplitude modulation using matrix values
    amplitude = np.repeat(flat_values, samp_per_ele)
    wave = amplitude * carrier
    
    return wave, t

def vec_wave(vector, matrix, normalize=True, samp_per_ele=100):
    """
    Convert a vector into a sinusoidal wave representation.
    
    Args:
        vector (list or np.array): Input vector
        normalize (bool): Whether to normalize values to [0, 2π] range
    
    Returns:
        np.array: 1D array of sinusoidal wave values
        np.array: Time/position array
    """
    # Convert to numpy array
    vector = np.array(vector)
    columns = matrix.shape[1]
    
    # Normalize to [0, 2π] range if requested
    if normalize:
        min_val = vector.min()
        max_val = vector.max()
        if max_val > min_val:
            normalized = (vector - min_val) / (max_val - min_val) * 2 * np.pi
    else:
        normalized = vector
    
    # Generate sinusoidal wave
    t = np.linspace(0, len(normalized), len(normalized) * samp_per_ele * columns)
    wave = np.zeros(len(t))
    
    # Create wave by summing sinusoids at each vector position
    for i, phase in enumerate(normalized):
        frequency = (i + 1) / len(normalized)  # Varying frequency
        wave += np.sin(2 * np.pi * frequency * t + phase)
    
    return wave, t

def visualize_matrix_wave(matrix, method='both'):
    """
    Visualize the matrix and its wave representation.
    
    Args:
        matrix (list or np.array): Input matrix
        method (str): 'amplitude', 'frequency', or 'both'
    """
    matrix = np.array(matrix)
    
    if method == 'both':
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original matrix
    im = ax1.imshow(matrix, cmap='viridis', aspect='auto')
    ax1.set_title(f'Original Matrix ({matrix.shape[0]}x{matrix.shape[1]})')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im, ax=ax1)
    
    # Plot wave(s)
    if method == 'amplitude':
        wave, t = AM(matrix)
        ax2.plot(t, wave)
        ax2.set_title('Amplitude-Modulated Sinusoidal Wave')
        ax2.set_xlabel('Time/Position')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
    elif method == 'frequency':
        wave, t = FSK(matrix)
        ax2.plot(t, wave)
        ax2.set_title('Frequency-Modulated Sinusoidal Wave')
        ax2.set_xlabel('Time/Position')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
    else:  # both
        wave_amp, t_amp = AM(matrix)
        ax2.plot(t_amp, wave_amp, color='blue', label='Amplitude Modulation')
        ax2.set_title('Amplitude-Modulated Sinusoidal Wave')
        ax2.set_xlabel('Time/Position')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        wave_phase, t_phase = FSK(matrix)
        ax3.plot(t_phase, wave_phase, color='red', label='Frequency Modulation')
        ax3.set_title('Frequency-Modulated Sinusoidal Wave')
        ax3.set_xlabel('Time/Position')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    test_matrix = [
        [5, 10, 15],
        [8, 12, 18],
        [3, 7, 20]
    ]
    
    print("Original Matrix:")
    for row in test_matrix:
        print(row)
    
    print("\nGenerating waves...")
    
    # Amplitude modulation
    wave_amp, t_amp = AM(test_matrix)
    print(f"Amplitude-modulated wave shape: {wave_amp.shape}")
    
    # Frequency modulation
    wave_phase, t_phase = FSK(test_matrix)
    print(f"Frequency-modulated wave shape: {wave_phase.shape}")
    
    # Visualize both waves
    visualize_matrix_wave(test_matrix, method='both')


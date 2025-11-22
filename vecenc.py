import numpy as np
import matplotlib.pyplot as plt


def fsk_encode(vector, samp_per_symbol=100, freq_min=1, freq_max=10):
    """
    Frequency Shift Keying (FSK) encoding of a vector.
    Each vector element maps to a unique frequency.
    
    Args:
        vector (list or np.array): Input vector
        samp_per_symbol (int): Samples per symbol/element
        freq_min (float): Minimum frequency (Hz)
        freq_max (float): Maximum frequency (Hz)
    
    Returns:
        np.array: FSK encoded wave signal
        np.array: Time array
    """
    vector = np.array(vector)
    
    # Normalize vector values to frequency range
    min_val = vector.min()
    max_val = vector.max()
    if max_val > min_val:
        frequencies = freq_min + (vector - min_val) / (max_val - min_val) * (freq_max - freq_min)
    else:
        frequencies = np.full_like(vector, (freq_min + freq_max) / 2, dtype=float)
    
    # Generate time array
    total_samples = len(vector) * samp_per_symbol
    t = np.linspace(0, len(vector), total_samples)
    
    # Generate FSK signal
    signal = np.zeros(total_samples)
    for i, freq in enumerate(frequencies):
        start_idx = i * samp_per_symbol
        end_idx = (i + 1) * samp_per_symbol
        t_segment = t[start_idx:end_idx]
        signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * t_segment)
    
    return signal, t


def qam_encode(vector, samp_per_symbol=100, carrier_freq=5):
    """
    Quadrature Amplitude Modulation (QAM) encoding of a vector.
    Encodes vector values into both amplitude and phase of carrier wave.
    
    Args:
        vector (list or np.array): Input vector
        samp_per_symbol (int): Samples per symbol/element
        carrier_freq (float): Carrier frequency (Hz)
    
    Returns:
        np.array: QAM encoded wave signal
        np.array: Time array
    """
    vector = np.array(vector)
    
    # Normalize vector to [-1, 1] range
    min_val = vector.min()
    max_val = vector.max()
    if max_val > min_val:
        normalized = 2 * (vector - min_val) / (max_val - min_val) - 1
    else:
        normalized = np.zeros_like(vector, dtype=float)
    
    # Split into I (in-phase) and Q (quadrature) components
    # Alternate values between I and Q channels
    I = np.zeros(len(vector))
    Q = np.zeros(len(vector))
    
    I[::2] = normalized[::2]  # Even indices to I
    if len(vector) > 1:
        Q[1::2] = normalized[1::2]  # Odd indices to Q
    
    # Generate time array
    total_samples = len(vector) * samp_per_symbol
    t = np.linspace(0, len(vector), total_samples)
    
    # Generate QAM signal: I*cos(ωt) - Q*sin(ωt)
    signal = np.zeros(total_samples)
    for i in range(len(vector)):
        start_idx = i * samp_per_symbol
        end_idx = (i + 1) * samp_per_symbol
        t_segment = t[start_idx:end_idx]
        
        # QAM modulation
        signal[start_idx:end_idx] = (I[i] * np.cos(2 * np.pi * carrier_freq * t_segment) - 
                                     Q[i] * np.sin(2 * np.pi * carrier_freq * t_segment))
    
    return signal, t


def visualize_vector_encoding(vector):
    """
    Visualize the vector and both encoding methods.
    
    Args:
        vector (list or np.array): Input vector
    """
    vector = np.array(vector)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original vector
    ax1.stem(range(len(vector)), vector, basefmt=' ')
    ax1.set_title(f'Original Vector (Length: {len(vector)})')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot FSK encoding
    fsk_signal, t_fsk = fsk_encode(vector)
    ax2.plot(t_fsk, fsk_signal, color='blue', linewidth=0.8)
    ax2.set_title('FSK (Frequency Shift Keying) Encoding')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot QAM encoding
    qam_signal, t_qam = qam_encode(vector)
    ax3.plot(t_qam, qam_signal, color='red', linewidth=0.8)
    ax3.set_title('QAM (Quadrature Amplitude Modulation) Encoding')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    test_vector = [3, 7, 2, 15, 9, 12, 5, 18, 10, 6]
    
    print("Original Vector:")
    print(test_vector)
    
    print("\nEncoding vector...")
    
    # FSK encoding
    fsk_signal, t_fsk = fsk_encode(test_vector)
    print(f"FSK signal shape: {fsk_signal.shape}")
    
    # QAM encoding
    qam_signal, t_qam = qam_encode(test_vector)
    print(f"QAM signal shape: {qam_signal.shape}")
    
    # Visualize both encodings
    visualize_vector_encoding(test_vector)

import numpy as np
from mpmath import mp

mp.dps = 60  # Set high precision decimal places for logistic map

def generate_logistic_noise(length, x0=0.7, mu=3.99):
    """
    Generate a pseudo-random noise mask using logistic map.
    
    Args:
        length (int): Number of bytes to generate.
        x0 (float): Initial value for logistic map (0 < x0 < 1).
        mu (float): Logistic map parameter (commonly near 3.99 for chaos).
        
    Returns:
        np.ndarray: Array of uint8 noise bytes.
    """
    x = mp.mpf(x0)
    mu = mp.mpf(mu)
    noise = []

    for _ in range(length):
        x = mu * x * (1 - x)
        byte = int((x % 1) * 256)  # Scale value to 0-255 range
        noise.append(byte)

    return np.array(noise, dtype=np.uint8)

def apply_noise_mask(plaintext_bytes, noise_mask):
    """
    Add logistic noise mask to plaintext bytes modulo 256.
    
    Args:
        plaintext_bytes (bytes): Original plaintext.
        noise_mask (np.ndarray): Noise mask bytes of same or greater length.
        
    Returns:
        bytes: Masked plaintext bytes.
    """
    plaintext_array = np.frombuffer(plaintext_bytes, dtype=np.uint8)
    masked_array = (plaintext_array + noise_mask[:len(plaintext_array)]) % 256
    return masked_array.astype(np.uint8).tobytes()

def remove_noise_mask(masked_bytes, noise_mask):
    """
    Remove logistic noise mask from masked bytes modulo 256.
    
    Args:
        masked_bytes (bytes): Masked plaintext bytes.
        noise_mask (np.ndarray): Noise mask bytes used during masking.
        
    Returns:
        bytes: Original plaintext bytes.
    """
    masked_array = np.frombuffer(masked_bytes, dtype=np.uint8)
    original_array = (masked_array - noise_mask[:len(masked_array)] + 256) % 256
    return original_array.astype(np.uint8).tobytes()

# === Example usage ===
if __name__ == "__main__":
    plaintext = b"Hello, this is a test message to verify the virtual noise-assisted encryption!"
    print(f"Original plaintext: {plaintext}")

    # Generate noise mask of same length as plaintext
    noise_mask = generate_logistic_noise(len(plaintext))
    print(f"Generated noise mask (first 16 bytes): {noise_mask[:16]}")

    # Apply noise mask
    masked_plaintext = apply_noise_mask(plaintext, noise_mask)
    print(f"Masked plaintext bytes: {masked_plaintext}")

    # Remove noise mask to recover original
    recovered_plaintext = remove_noise_mask(masked_plaintext, noise_mask)
    print(f"Recovered plaintext: {recovered_plaintext}")

    # Verify correctness
    assert recovered_plaintext == plaintext, "Error: recovered plaintext does not match original!"
    print("Success: Recovered plaintext matches original!")

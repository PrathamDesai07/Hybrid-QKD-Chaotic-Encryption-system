import numpy as np
from mpmath import mp

# Set high-precision arithmetic for chaos map
mp.dps = 60  # Decimal places for mpmath precision

def generate_logistic_noise(length, x0=0.7, mu=3.99):
    """
    Generate a pseudo-random noise mask using the logistic map (chaotic system).

    Args:
        length (int): Number of bytes to generate.
        x0 (float): Initial seed value (0 < x0 < 1).
        mu (float): Logistic map parameter (typically near 3.99).

    Returns:
        np.ndarray: Array of uint8 values forming the noise mask.
    """
    length = int(length)
    if length <= 0:
        return np.array([], dtype=np.uint8)

    if not (0 < x0 < 1):
        x0 = 0.5  # fallback to neutral mid-range seed

    x = mp.mpf(x0)
    mu = mp.mpf(mu)
    noise = []

    for _ in range(length):
        x = mu * x * (1 - x)
        byte = int((x % 1) * 256) % 256  # convert to [0, 255]
        noise.append(byte)

    return np.array(noise, dtype=np.uint8)

def apply_noise_mask(plaintext_bytes, noise_mask):
    """
    Apply logistic chaos-based noise to the plaintext bytes.

    Args:
        plaintext_bytes (bytes): Data to mask.
        noise_mask (np.ndarray): Pre-generated noise mask.

    Returns:
        bytes: Encrypted/masked data.
    """
    if noise_mask is None or len(noise_mask) == 0:
        raise ValueError("Noise mask is empty. Ensure valid input seed and length.")

    plaintext_array = np.frombuffer(plaintext_bytes, dtype=np.uint8)
    masked_array = (plaintext_array + noise_mask[:len(plaintext_array)]) % 256
    return masked_array.astype(np.uint8).tobytes()

def remove_noise_mask(masked_bytes, noise_mask):
    """
    Reverse logistic chaos-based noise from the masked bytes.

    Args:
        masked_bytes (bytes): Encrypted or masked data.
        noise_mask (np.ndarray): The same noise mask used to encrypt.

    Returns:
        bytes: Recovered original data.
    """
    if noise_mask is None or len(noise_mask) == 0:
        raise ValueError("Noise mask is empty. Cannot reverse masking.")

    masked_array = np.frombuffer(masked_bytes, dtype=np.uint8)
    original_array = (masked_array - noise_mask[:len(masked_array)] + 256) % 256
    return original_array.astype(np.uint8).tobytes()

# === Example test (runs only when script is standalone) ===
if __name__ == "__main__":
    plaintext = b"Hello, this is a test message to verify the virtual noise-assisted encryption!"
    print(f"Original plaintext: {plaintext}")

    x0_seed = 0.71342812  # Example valid seed
    noise_mask = generate_logistic_noise(len(plaintext), x0=x0_seed)

    print(f"Generated noise mask (first 16 bytes): {noise_mask[:16]}")
    masked = apply_noise_mask(plaintext, noise_mask)
    print(f"Masked bytes: {masked}")

    recovered = remove_noise_mask(masked, noise_mask)
    print(f"Recovered plaintext: {recovered}")

    assert recovered == plaintext, "❌ Error: Recovered plaintext does not match original!"
    print("✅ Success: Recovered plaintext matches original.")

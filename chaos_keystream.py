import numpy as np
from mpmath import mp
from scipy.integrate import solve_ivp
from math import floor 


# Set mpmath precision (60 decimal places)
mp.dps = 60

# Utility: Convert bits array to float in (0, 1)
def bits_to_float(bits):
    bit_str = ''.join(map(str, bits))
    int_val = int(bit_str, 2)
    max_val = 2 ** len(bits)
    return int_val / max_val

# Map selectors codes:
# 00 = logistic, 01 = lorenz, 10 = chen, 11 = custom (fallback to logistic)
MAP_CODES = {
    '00': 'logistic',
    '01': 'lorenz',
    '10': 'chen',
    '11': 'logistic'
}

# --- Logistic Map with mpmath (high precision) ---
def logistic_map(x0, mu, n_iters):
    x = mp.mpf(x0)
    mu = mp.mpf(mu)
    sequence = []
    for _ in range(n_iters):
        x = mu * x * (1 - x)
        sequence.append(x)
    return sequence

# --- Chen System ODE ---
def chen_ode(t, state, a=35, b=3, c=28):
    x, y, z = state
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return [dx, dy, dz]

# --- Lorenz System ODE ---
def lorenz_ode(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# --- Quantization Helpers ---

def quantize_logistic_chen(sequence, bits_per_iter=4):
    """
    For logistic and chen maps, extract least significant bits
    from fractional decimal part (mpf).
    """
    bits = []
    for val in sequence:
        # val is mpmath float in (0,1)
        # Multiply by 2^bits_per_iter and floor to integer
        scaled = int(floor(val * (2 ** bits_per_iter)))
        # Extract bits_per_iter bits from scaled
        for i in reversed(range(bits_per_iter)):
            bits.append((scaled >> i) & 1)
    return bits

def quantize_lorenz(x_values, bits_per_value=16):
    """
    Discretize Lorenz x-coordinate to 16-bit integers.
    Normalize x in some range, then convert.
    """
    bits = []
    x_min, x_max = min(x_values), max(x_values)
    range_span = x_max - x_min if x_max != x_min else 1.0
    for x in x_values:
        # Normalize x to [0,1]
        norm_x = (x - x_min) / range_span
        int_val = int(norm_x * (2**bits_per_value - 1))
        for i in reversed(range(bits_per_value)):
            bits.append((int_val >> i) & 1)
    return bits

# --- Keystream generation pipeline ---
def generate_keystream_from_final_key(final_key_bits, keystream_length=1024):
    """
    Generate keystream bits by:
    1) Selecting chaotic map from first 2 bits
    2) Extracting seeds from next bits
    3) Running chaotic simulation with high precision
    4) Quantizing chaotic states to bits
    5) Pooling bits until desired length is reached
    """
    # --- 1) Map selection ---
    map_bits = ''.join(map(str, final_key_bits[:2]))
    chaotic_map = MAP_CODES.get(map_bits, 'logistic')
    print(f"Selected chaotic map: {chaotic_map}")

    # --- 2) Seed extraction ---
    # For logistic map: need 1 seed (64 bits)
    # For Lorenz and Chen: need 3 seeds (x,y,z), 64 bits each
    if chaotic_map == 'logistic':
        seed_bits = final_key_bits[2:66]
        seed = bits_to_float(seed_bits)
        mu = 3.99  # chaotic parameter for logistic map near fully chaotic regime
        print(f"Logistic seed: {seed:.15f}, mu: {mu}")

        # Run logistic map for enough iterations to get keystream bits
        n_iters = int(np.ceil(keystream_length / 4))  # 4 bits per iter
        seq = logistic_map(seed, mu, n_iters)

        # Quantize output to bits
        bits = quantize_logistic_chen(seq, bits_per_iter=4)

    else:
        # For Lorenz/Chen extract 3 seeds (64 bits each)
        seed_x_bits = final_key_bits[2:66]
        seed_y_bits = final_key_bits[66:130]
        seed_z_bits = final_key_bits[130:194]
        seed_x = bits_to_float(seed_x_bits) * 30  # scale approx range of states
        seed_y = bits_to_float(seed_y_bits) * 30
        seed_z = bits_to_float(seed_z_bits) * 30
        initial_state = [seed_x, seed_y, seed_z]
        print(f"Initial state x,y,z: {seed_x:.5f}, {seed_y:.5f}, {seed_z:.5f}")

        # Integration time and steps (enough iterations)
        t_span = (0, 50)
        max_step = 0.01

        if chaotic_map == 'lorenz':
            sol = solve_ivp(lorenz_ode, t_span, initial_state, max_step=max_step, rtol=1e-9)
            x_vals = sol.y[0]
            bits = quantize_lorenz(x_vals, bits_per_value=16)

        elif chaotic_map == 'chen':
            sol = solve_ivp(chen_ode, t_span, initial_state, max_step=max_step, rtol=1e-9)
            # Chen system is continuous like Lorenz - quantize x coordinate similarly
            x_vals = sol.y[0]
            bits = quantize_lorenz(x_vals, bits_per_value=16)

        else:
            raise ValueError("Unsupported map selected")

    # --- 5) Pool bits to requested keystream length ---
    if len(bits) < keystream_length:
        raise ValueError(f"Generated keystream too short: {len(bits)} bits < {keystream_length}")

    keystream = bits[:keystream_length]
    print(f"Generated keystream length: {len(keystream)} bits")
    return np.array(keystream, dtype=np.uint8)

# --- Load final key and run ---
if __name__ == "__main__":
    # Load 256-bit final key from saved .npy file (assumed from BB84 pipeline)
    final_key = np.load("final_key.npy")
    print(f"Final key length: {len(final_key)} bits")

    # Generate 1024-bit keystream from final key
    keystream = generate_keystream_from_final_key(final_key, keystream_length=1024)

    # Save the keystream for downstream classical encryption
    np.save("keystream.npy", keystream)
    print("Keystream saved to keystream.npy")

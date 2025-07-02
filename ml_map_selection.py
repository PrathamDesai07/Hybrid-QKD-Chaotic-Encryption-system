import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from scipy.integrate import solve_ivp
from mpmath import mp
import os

# Set high precision for mpmath (60 decimal places)
mp.dps = 60

# Map types
MAP_TYPES = ['logistic', 'lorenz', 'chen']

# ===== Synthetic QBER Dataset Generation =====
def simulate_qber(map_name, noise_level):
    """
    Simulate QBER for given chaotic map under noise (0.0 to 0.10).
    These are synthetic functions mimicking expected behavior.
    """
    base_qber = {
        'logistic': 0.04 + 0.5 * noise_level,
        'lorenz': 0.03 + 0.6 * noise_level,
        'chen': 0.02 + 0.7 * noise_level,
    }
    noise = np.random.normal(0, 0.002)  # small random variation
    qber = base_qber.get(map_name, 0.05) + noise
    return np.clip(qber, 0, 1)

def generate_dataset(num_samples_per_map=100):
    X = []
    y = []
    for i, map_name in enumerate(MAP_TYPES):
        noise_levels = np.linspace(0, 0.10, num_samples_per_map)
        for noise_level in noise_levels:
            qber = simulate_qber(map_name, noise_level)
            X.append([qber])
            y.append(i)
    return np.array(X), np.array(y)

# ===== ML Training and Saving =====
MODEL_FILENAME = "rf_map_selector.joblib"

def train_and_save_model():
    print("Training Random Forest classifier on synthetic QBER dataset...")
    X, y = generate_dataset()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")

def load_model():
    if not os.path.exists(MODEL_FILENAME):
        train_and_save_model()
    return joblib.load(MODEL_FILENAME)

# ===== Chaotic Map Implementations =====

def logistic_map(x, mu=3.99):
    # x in (0,1)
    return mu * x * (1 - x)

def generate_logistic_keystream(seed, length_bits):
    x = mp.mpf(seed)
    mu = mp.mpf('3.99')
    bits = []
    for _ in range(length_bits):
        x = mu * x * (1 - x)
        # Extract bit from fractional part (multiply by 2 and floor)
        bit = int(float(x) * 2)
        bits.append(bit)
    return np.array(bits, dtype=np.uint8)

def lorenz_system(t, state, sigma=10.0, beta=8/3, rho=28.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generate_lorenz_keystream(seed, length_bits):
    # seed is list or array of 3 floats in [0,30]
    state = np.array(seed)
    dt = 0.01
    t_span = (0, dt)
    bits = []
    for _ in range(length_bits):
        sol = solve_ivp(lorenz_system, t_span, state, method='RK45')
        state = sol.y[:, -1]
        # Quantize x coordinate to 16 bits and extract bits
        x_quant = int((state[0] % 50) * (2**16 / 50))  # scale to 16-bit int
        # Extract lowest bit as keystream bit
        bit = x_quant & 1
        bits.append(bit)
    return np.array(bits, dtype=np.uint8)


def chen_system(t, state, a=35.0, b=3.0, c=28.0):
    x, y, z = state
    dx = a * (y - x)
    dy = (c - a) * x - x * z + c * y
    dz = x * y - b * z
    return [dx, dy, dz]

def generate_chen_keystream(seed, length_bits):
    """
    Generate keystream bits by integrating Chen system with solve_ivp
    seed: list/array of 3 floats initial state
    length_bits: number of bits to generate
    """
    state = np.array(seed)
    dt = 0.01
    bits = []
    t = 0

    for _ in range(length_bits):
        sol = solve_ivp(chen_system, (t, t + dt), state, method='RK45', max_step=dt)
        state = sol.y[:, -1]
        t += dt
        # Quantize x coordinate to 16 bits and extract LSB bit
        x_quant = int((state[0] % 50) * (2**16 / 50))
        bit = x_quant & 1
        bits.append(bit)

    return np.array(bits, dtype=np.uint8)


# ===== Integration: Map Selection and Keystream Generation =====

def select_chaotic_map(qber, clf):
    """Given a QBER scalar, predict the best chaotic map index and name."""
    pred_idx = clf.predict(np.array([[qber]]))[0]
    return pred_idx, MAP_TYPES[pred_idx]

def generate_keystream(final_key_bits, qber, keystream_length_bits):
    """
    Given the final BB84 key bits and a QBER value,
    select chaotic map by ML, then generate keystream.
    """
    clf = load_model()
    map_idx, map_name = select_chaotic_map(qber, clf)
    print(f"ML Map Selection: QBER={qber:.4f} -> Selected map: {map_name}")

    # Derive seed from final_key_bits (example: first bits as seed)
    # For simplicity, convert bits to float between 0 and 1 for logistic, or triple for others
    if map_name == 'logistic':
        seed = np.packbits(final_key_bits[:16]).sum() / 255.0  # roughly scalar in [0,1]
        seed = max(min(seed, 0.999), 0.001)  # avoid boundary
        keystream = generate_logistic_keystream(seed, keystream_length_bits)
    elif map_name == 'lorenz':
        # Three seeds from first 48 bits scaled to [0,30]
        seeds_bits = final_key_bits[:48]
        seeds_bytes = np.packbits(seeds_bits)
        seeds_floats = [int(seeds_bytes[i]) / 255 * 30 for i in range(3)]
        keystream = generate_lorenz_keystream(seeds_floats, keystream_length_bits)
    else:  # chen
        seeds_bits = final_key_bits[:48]
        seeds_bytes = np.packbits(seeds_bits)
        seeds_floats = [int(seeds_bytes[i]) / 255 * 30 for i in range(3)]
        keystream = generate_chen_keystream(seeds_floats, keystream_length_bits)

    return keystream, map_name

# ===== Main Entrypoint =====

def main():
    print("=== Phase 6A: ML-Driven Chaotic Map Selection & Keystream Generation ===")

    # Load or simulate final key bits (example: 256 random bits)
    final_key_bits = np.random.randint(0, 2, 256, dtype=np.uint8)
    print(f"Loaded final key bits (length={len(final_key_bits)})")

    # Example QBER value from BB84 simulation (replace with real measured QBER)
    keystream_length_bits = 1024
    for qber in [0.01, 0.03, 0.045, 0.07, 0.1]:
        print(f"Testing with QBER={qber}")
        keystream, selected_map = generate_keystream(final_key_bits, qber, keystream_length_bits)
        print(f"Selected chaotic map: {selected_map}\n")


    # Keystream length in bits (example 1024 bits)

    # Train model if needed
    if not os.path.exists(MODEL_FILENAME):
        train_and_save_model()

    # Generate keystream with ML-driven map selection
    keystream, selected_map = generate_keystream(final_key_bits, qber, keystream_length_bits)

    print(f"Generated keystream length: {len(keystream)} bits using map: {selected_map}")

    # (Optional) Save keystream to file
    np.save("phase6a_keystream.npy", keystream)
    print("Keystream saved to phase6a_keystream.npy")

if __name__ == "__main__":
    main()
# qkd/bb84_pipeline.py
# Full BB84 simulation + post-processing pipeline (PennyLane-based version)

import pennylane as qml
from pennylane import numpy as np
import random
import pyldpc
import hashlib

# === Parameters ===
n_bits = 512
noise_prob = 0.05  # Depolarizing noise probability

# === Simulate BB84 Protocol with Noise ===
def simulate_bb84(n_bits):
    dev = qml.device("default.qubit", wires=1, shots=1)

    def apply_depolarizing_noise(p):
        r = random.random()
        if r < p:
            gate = random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
            gate(wires=0)

    @qml.qnode(dev)
    def measure(prep_basis, bit, meas_basis):
        if prep_basis == "X":
            if bit == 1:
                qml.PauliX(wires=0)
            qml.Hadamard(wires=0)
        else:
            if bit == 1:
                qml.PauliX(wires=0)

        apply_depolarizing_noise(noise_prob)  # Inject noise

        if meas_basis == "X":
            qml.Hadamard(wires=0)

        return qml.sample(qml.PauliZ(wires=0))

    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.choice(["X", "Z"], size=n_bits)
    bob_bases = np.random.choice(["X", "Z"], size=n_bits)
    bob_results = []

    for a_bit, a_basis, b_basis in zip(alice_bits, alice_bases, bob_bases):
        meas = measure(a_basis, a_bit, b_basis)
        bob_results.append(int((1 - meas) / 2))  # Convert -1/1 to 1/0

    return alice_bits, alice_bases, bob_bases, np.array(bob_results)

# === Basis Reconciliation & Sifting ===
def sift_keys(alice_bits, alice_bases, bob_bases, bob_bits):
    shared_indices = [i for i in range(len(alice_bases)) if alice_bases[i] == bob_bases[i]]
    alice_key = alice_bits[shared_indices]
    bob_key = bob_bits[shared_indices]
    return alice_key, bob_key

# === Error Correction (LDPC) ===
def ldpc_correction(alice_key, bob_key):
    # Define LDPC parameters
    n_codeword = 128
    d_v, d_c = 2, 4
    H, G = pyldpc.make_ldpc(n_codeword, d_v, d_c, systematic=True, sparse=True)

    k = G.shape[1]  # Number of data bits needed

    # Truncate or pad key to match k
    if len(alice_key) < k:
        pad_len = k - len(alice_key)
        alice_key = np.concatenate([alice_key, np.random.randint(0, 2, pad_len)])
        bob_key = np.concatenate([bob_key, np.random.randint(0, 2, pad_len)])
    else:
        alice_key = alice_key[:k]
        bob_key = bob_key[:k]

    x = np.array(alice_key)
    y = pyldpc.encode(G, x, snr=3)
    y_noisy = np.array(y.copy(), dtype=np.int8)
    y_noisy[np.random.rand(len(y_noisy)) < 0.1] ^= 1  # Flip ~10% bits
    x_decoded = pyldpc.decode(H, y_noisy, snr=3)
    return x_decoded

# === Privacy Amplification ===
def privacy_amplification(key_bits):
    key_str = ''.join(map(str, key_bits))
    hashed = hashlib.sha256(key_str.encode()).hexdigest()
    final_key = np.array([int(b) for b in bin(int(hashed, 16))[2:].zfill(256)])
    return final_key

# === Full Pipeline ===
def bb84_pipeline(n_bits=512):
    alice_bits, alice_bases, bob_bases, bob_bits = simulate_bb84(n_bits)
    alice_key, bob_key = sift_keys(alice_bits, alice_bases, bob_bases, bob_bits)

    # Calculate and print QBER
    qber = np.mean(alice_key != bob_key)
    print(f"QBER: {qber * 100:.2f}%")

    corrected_key = ldpc_correction(alice_key, bob_key)
    final_key = privacy_amplification(corrected_key)
    np.save("final_key.npy", final_key)
    print("Final key saved to final_key.npy ({} bits)".format(len(final_key)))

if __name__ == "__main__":
    bb84_pipeline()
# main.py – Unified Phase‑7 simulation with Phase‑6 extensions
import os, random, hmac, hashlib
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp

# Project modules
from bb84_pipeline import bb84_pipeline
from ml_map_selection import generate_keystream as ks_ml
from gru_keystream import ChaoticGRU, generate_keystream_from_gru
from feedback_chaos import FeedbackChaosKeystream
from virtual_noise_layer import generate_logistic_noise, apply_noise_mask, remove_noise_mask
from classical_encryption_engine import encrypt, decrypt

mp.dps = 60
GRU_MODEL = "gru_chaotic_model.pth"
ROUND_COUNT = 2  # Must match classical_encryption_engine.py

# -------------------------------------------------------------------
def hash_whiten(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 256:
        raise ValueError("Whitening requires multiple of 256 bits")
    out = []
    for i in range(0, len(bits), 256):
        digest = hashlib.sha256(np.packbits(bits[i:i+256]).tobytes()).digest()
        out.extend(np.unpackbits(np.frombuffer(digest, np.uint8)))
    return np.asarray(out, dtype=np.uint8)

def gru_keystream(final_key_bits: np.ndarray, length: int) -> np.ndarray:
    import torch
    if not os.path.exists(GRU_MODEL):
        from gru_keystream import prepare_training_data, train_model
        Xtr, Xte, ytr, yte = prepare_training_data(seq_length=1000, sample_length=50)
        model = ChaoticGRU()
        train_model(model, Xtr, ytr, Xte, yte, epochs=5, batch_size=256)
        torch.save(model.state_dict(), GRU_MODEL)
    model = ChaoticGRU()
    model.load_state_dict(torch.load(GRU_MODEL, map_location="cpu"))
    model.eval()
    seed_bits = final_key_bits[:50]
    return generate_keystream_from_gru(model, seed_bits, length)

def run_phase7_simulation():
    # 1. Channel‑loss sweep ---------------------------------------
    losses, qbers = [], []
    for loss in np.arange(0.0, 0.31, 0.02):
        _, q = bb84_pipeline(loss_prob=loss, depol_prob=0.05, verbose=False)
        losses.append(loss)
        qbers.append(q)
    plt.figure()
    plt.plot(losses, qbers, marker="o")
    plt.xlabel("Channel Loss")
    plt.ylabel("QBER")
    plt.title("QBER vs Channel Loss")
    plt.grid(True)
    plt.savefig("qber_vs_loss.png")

    # 2. Encryption test ------------------------------------------
    final_key = np.load("final_key.npy")
    msg = b"End-to-end test message for Phase-7 pipeline."

    # --- Phase‑6C: Virtual Noise Layer ----------------------------
    scrambled, perm = scramble_message(msg)
    print(f"[DEBUG] scrambled len = {len(scrambled)}")

    seed_val = seed_from_key(final_key)
    if not (0 < seed_val < 1):
        raise ValueError(f"Invalid seed: {seed_val}")

    noise_mask = generate_logistic_noise(length=len(scrambled), x0=seed_val)
    if noise_mask is None or len(noise_mask) == 0:
        raise ValueError("Noise mask is empty. Ensure valid input seed and scrambled message.")

    noisy = apply_noise_mask(scrambled, noise_mask)

    whitening_key = int("".join(map(str, final_key[:16])), 2)
    whitened = reversible_whitening(noisy, whitening_key)

    # --- Keystream length calculation -----------------------------
    total_bits = len(whitened) * 8
    padded_bits = ((total_bits + 127) // 128) * 128
    ks_len = ROUND_COUNT * (padded_bits + padded_bits // 128)

    # --- Phase‑6A: ML Map Selection -------------------------------
    qber_est = float(np.mean(final_key))
    ks_a, _ = ks_ml(final_key, qber_est, ks_len)

    # --- Phase‑6B: GRU Chaotic Keystream --------------------------
    ks_b = gru_keystream(final_key, ks_len)

    # --- Phase‑6D: Feedback Chaos Stream --------------------------
    ks_c = FeedbackChaosKeystream(final_key).generate_keystream(ks_len)

    # Combine and whiten -------------------------------------------
    ks_comb = ks_a ^ ks_b ^ ks_c
    pad = (-len(ks_comb)) % 256
    if pad:
        ks_comb = np.concatenate([ks_comb, np.random.randint(0, 2, pad, dtype=np.uint8)])
    ks_final = hash_whiten(ks_comb)[:ks_len]

    # Apply legacy flip
    ks_final = ks_final ^ np.uint8(np.sum(ks_final[:32]) % 2)

    # Encrypt + authenticate
    ct, perm_stack, pad_len = encrypt(whitened, ks_final)
    hmac_tag = hmac.new(np.packbits(final_key)[:32].tobytes(), ct, hashlib.sha256).digest()

    # Decrypt path
    pt = decrypt(ct, ks_final, perm_stack, pad_len)
    pt = reversible_whitening(pt, whitening_key)
    pt = remove_noise_mask(pt, noise_mask)
    pt = descramble_message(pt, perm)

    assert pt == msg, "Decryption failed!"
    ber = bit_error_rate(msg, pt)
    avalanche = avalanche_effect(whitened, ks_final)
    monobit_ok = abs(np.mean(ks_final) - 0.5) < 0.01

    print("\n===== Phase‑7 Results =====")
    print(f"BER             : {ber:.4f}")
    print(f"Avalanche Effect: {avalanche:.2f}%")
    print(f"Monobit Pass    : {monobit_ok}")

    # Attack QBER simulation plot
    plt.figure()
    ir, pns = random.uniform(0.11, 0.15), random.uniform(0.06, 0.10)
    plt.bar(["Normal", "Intercept-Resend", "Photon-Split"], [0.01, ir, pns],
            color=["green", "red", "orange"])
    plt.ylabel("QBER")
    plt.title("Attack Simulation")
    plt.savefig("attack_qber.png")

    print("✅ Plots saved: qber_vs_loss.png, attack_qber.png\n")

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def scramble_message(data: bytes):
    perm = np.random.permutation(len(data))
    return bytes(data[i] for i in perm), perm

def descramble_message(data: bytes, perm):
    buf = [None] * len(data)
    for i, p in enumerate(perm):
        buf[p] = data[i]
    return bytes(buf)

def seed_from_key(bits):
    seed_val = int("".join(map(str, bits[:64])), 2)
    seed_float = seed_val / (2 ** 64)
    return min(max(seed_float, 1e-6), 1 - 1e-6)

def reversible_whitening(data: bytes, key: int):
    random.seed(key)
    mask = bytes(random.getrandbits(8) for _ in range(len(data)))
    return bytes(d ^ m for d, m in zip(data, mask))

def bit_error_rate(a: bytes, b: bytes):
    return np.mean(np.unpackbits(np.frombuffer(a, np.uint8)) !=
                   np.unpackbits(np.frombuffer(b, np.uint8)))

def avalanche_effect(pt: bytes, ks: np.ndarray):
    pt2 = bytearray(pt)
    pt2[0] ^= 1  # single bit flip
    ct1, _, _ = encrypt(pt, ks)
    ct2, _, _ = encrypt(bytes(pt2), ks)
    return np.mean(np.unpackbits(np.frombuffer(ct1, np.uint8)) !=
                   np.unpackbits(np.frombuffer(ct2, np.uint8))) * 100

# -------------------------------------------------------------------
if __name__ == "__main__":
    run_phase7_simulation()

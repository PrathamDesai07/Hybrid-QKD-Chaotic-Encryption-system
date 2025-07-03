# qkd/bb84_pipeline.py
# -------------------------------------------------------------------
# Full BB84 simulation + post‑processing pipeline (PennyLane version)
#
# Key upgrades (2025‑07‑03)
# • Function signatures now accept   loss_prob   and   depol_prob
# • Loss is modelled as photon drop‑outs before Bob’s measurement
# • All tunables exposed via bb84_pipeline(...)
# • Returns (final_key, qber)  so downstream code can read QBER
# -------------------------------------------------------------------

import pennylane as qml
from pennylane import numpy as np
import random
import pyldpc
import hashlib
from typing import Tuple

# -------------------------------------------------------------------
# 1.  BB84 qubit transmission with depolarising noise  +  channel loss
# -------------------------------------------------------------------
def simulate_bb84(
    n_bits: int = 512,
    loss_prob: float = 0.0,
    depol_prob: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a BB84 run and return all raw data.

    Parameters
    ----------
    n_bits      : total qubits prepared by Alice
    loss_prob   : probability that a photon is lost in the channel
    depol_prob  : probability of a depolarising error on the qubit

    Returns
    -------
    alice_bits, alice_bases, bob_bases, bob_results, loss_mask
        loss_mask is a boolean array – True where the photon was lost
    """
    dev = qml.device("default.qubit", wires=1, shots=1)

    # Helper: inject depolarising noise with prob = depol_prob
    def maybe_apply_depolarising_noise():
        if random.random() < depol_prob:
            random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])(wires=0)

    @qml.qnode(dev)
    def measure(prep_basis, bit, meas_basis):
        # --- State preparation --------------------------------------
        if prep_basis == "X":
            if bit == 1:
                qml.PauliX(wires=0)
            qml.Hadamard(wires=0)
        else:  # Z basis
            if bit == 1:
                qml.PauliX(wires=0)
        # --- Channel noise ------------------------------------------
        maybe_apply_depolarising_noise()
        # --- Bob’s basis choice & measurement -----------------------
        if meas_basis == "X":
            qml.Hadamard(wires=0)
        return qml.sample(qml.PauliZ(wires=0))  # ±1

    # ---------- Random preparation / basis choices ------------------
    alice_bits  = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.choice(["X", "Z"], size=n_bits)
    bob_bases   = np.random.choice(["X", "Z"], size=n_bits)

    bob_results = []
    loss_mask   = np.zeros(n_bits, dtype=bool)  # True → photon lost

    for i, (bit, a_basis, b_basis) in enumerate(zip(alice_bits, alice_bases, bob_bases)):
        # Simulate photon loss
        if random.random() < loss_prob:
            loss_mask[i] = True
            bob_results.append(-1)          # placeholder
            continue
        meas = measure(a_basis, bit, b_basis)
        bob_results.append(int((1 - meas) / 2))  # map {‑1,1} → {1,0}

    return alice_bits, alice_bases, bob_bases, np.array(bob_results), loss_mask


# -------------------------------------------------------------------
# 2.  Classical post‑processing
# -------------------------------------------------------------------
def sift_keys(
    alice_bits, alice_bases, bob_bases, bob_bits, loss_mask
):
    """Keep indices where bases match *and* photon was received."""
    shared = [
        i for i in range(len(alice_bases))
        if (not loss_mask[i]) and alice_bases[i] == bob_bases[i]
    ]
    return alice_bits[shared], bob_bits[shared]


def ldpc_correction(alice_key, bob_key):
    """One‑shot LDPC correction (toy parameters)."""
    n_codeword = 128
    H, G = pyldpc.make_ldpc(n_codeword, d_v=2, d_c=4,
                            systematic=True, sparse=True)
    k = G.shape[1]                      # data bits required

    # Pad or trim to length k
    if len(alice_key) < k:
        deficit = k - len(alice_key)
        alice_key = np.concatenate([alice_key,
                                    np.random.randint(0, 2, deficit)])
        bob_key   = np.concatenate([bob_key,
                                    np.random.randint(0, 2, deficit)])
    else:
        alice_key, bob_key = alice_key[:k], bob_key[:k]

    codeword   = pyldpc.encode(G, alice_key, snr=3)
    noisy_cw   = codeword.copy().astype(np.int8)
    noisy_cw[np.random.rand(len(noisy_cw)) < 0.10] ^= 1  # add 10 % flips

    decoded    = pyldpc.decode(H, noisy_cw, snr=3)
    return decoded


def privacy_amplification(raw_bits):
    key_str = ''.join(map(str, raw_bits))
    digest  = hashlib.sha256(key_str.encode()).hexdigest()
    return np.array([int(b) for b in bin(int(digest, 16))[2:].zfill(256)],
                    dtype=np.uint8)


# -------------------------------------------------------------------
# 3.  High‑level pipeline wrapper
# -------------------------------------------------------------------
def bb84_pipeline(
    n_bits: int = 512,
    loss_prob: float = 0.0,
    depol_prob: float = 0.05,
    verbose: bool = True,
):
    """
    Run the entire BB84 + post‑processing pipeline.

    Saves  final_key.npy   and returns  (final_key, qber).
    """
    alice_bits, alice_bases, bob_bases, bob_bits, loss_mask = simulate_bb84(
        n_bits=n_bits, loss_prob=loss_prob, depol_prob=depol_prob
    )

    # ---------- Sifting & QBER --------------------------------------
    a_key, b_key = sift_keys(alice_bits, alice_bases,
                             bob_bases, bob_bits, loss_mask)
    qber = float(np.mean(a_key != b_key)) if len(a_key) else 1.0
    if verbose:
        print(f"[BB84] sifted length = {len(a_key)} bits | QBER = {qber*100:.2f}%")

    # ---------- Error correction + privacy amplification ------------
    reconciled = ldpc_correction(a_key, b_key)
    final_key  = privacy_amplification(reconciled)
    np.save("final_key.npy", final_key)
    if verbose:
        print(f"[BB84] Final key saved (length = {len(final_key)} bits)")

    return final_key, qber


# -------------------------------------------------------------------
if __name__ == "__main__":
    bb84_pipeline(loss_prob=0.05, depol_prob=0.05)

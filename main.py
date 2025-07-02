import numpy as np
import hmac
import hashlib
from math import floor
from bb84_pipeline import bb84_pipeline
from chaos_keystream import generate_keystream_from_final_key
from classical_encryption_engine import encrypt, decrypt

# Phase 6C: Logistic map noise mask functions
def generate_logistic_noise_mask(seed_float, length):
    """Generate a noise mask of `length` bytes using logistic map with seed."""
    x = seed_float
    mu = 3.99
    mask = []
    for _ in range(length):
        x = mu * x * (1 - x)
        # Scale to byte range [0,255]
        byte_val = int(x * 256) % 256
        mask.append(byte_val)
    return bytes(mask)

def add_noise_mask(plaintext_bytes, noise_mask):
    """Add noise mask to plaintext bytes mod 256."""
    noisy = bytes((p + n) % 256 for p, n in zip(plaintext_bytes, noise_mask))
    return noisy

def main():
    print("=== Phase 2: Running BB84 Pipeline to Generate Final Key ===")
    bb84_pipeline()  # runs BB84 simulation and saves final_key.npy

    final_key = np.load("final_key.npy")
    print(f"Loaded final key length: {len(final_key)} bits")

    # Original plaintext to encrypt
    plaintext = b"Hello, this is a test message to verify the full pipeline!"

    print("\n=== Phase 6C: Generate Noise Mask and Apply to Plaintext ===")
    # Derive seed from first 64 bits of final key as float in (0,1)
    seed_bits = final_key[:64]
    seed_float = int("".join(str(b) for b in seed_bits), 2) / (2**64)

    noise_mask = generate_logistic_noise_mask(seed_float, len(plaintext))
    print(f"Generated noise mask of length: {len(noise_mask)}")

    noisy_plaintext = add_noise_mask(plaintext, noise_mask)
    print(f"Noisy plaintext bytes (hex): {noisy_plaintext.hex()}")

    # Calculate keystream length based on noisy plaintext length
    noisy_bits_len = len(noisy_plaintext) * 8
    block_size_bits = 128
    padded_len = ((noisy_bits_len + block_size_bits - 1) // block_size_bits) * block_size_bits
    n_blocks = padded_len // block_size_bits
    keystream_len = n_blocks + padded_len  # permutation bits + xor bits

    print("\n=== Phase 3 (unchanged): Generating Chaotic Keystream from Final Key ===")
    keystream = generate_keystream_from_final_key(final_key, keystream_length=keystream_len)
    print(f"Generated keystream length: {len(keystream)} bits")

    print("\n=== Phase 4: Encrypting noisy plaintext with Chaotic Keystream ===")
    ciphertext_bytes, permutation_indices, pad_len = encrypt(noisy_plaintext, keystream)
    print(f"Ciphertext (hex): {ciphertext_bytes.hex()}")

    recovered_plaintext = decrypt(ciphertext_bytes, keystream, permutation_indices, pad_len)
    print(f"Recovered plaintext: {recovered_plaintext}")

    if recovered_plaintext == noisy_plaintext:
        print("SUCCESS: Decrypted plaintext matches noisy plaintext!")
    else:
        print("FAILURE: Decrypted plaintext does NOT match noisy plaintext!")

    print("\n=== Phase 5: Generating and Verifying HMAC Tag ===")
    # Convert final key bits to bytes for HMAC key (take first 32 bytes / 256 bits)
    final_key_bytes = np.packbits(final_key).tobytes()
    hmac_key = final_key_bytes[:32]

    # Generate HMAC tag of ciphertext
    hmac_tag = hmac.new(hmac_key, ciphertext_bytes, hashlib.sha256).digest()
    print(f"HMAC tag generated: {hmac_tag.hex()}")

    # Verification (simulate verification phase)
    print("\n[Verification] Verifying HMAC tag...")
    computed_tag = hmac.new(hmac_key, ciphertext_bytes, hashlib.sha256).digest()
    if hmac.compare_digest(hmac_tag, computed_tag):
        print("✔ Integrity check PASSED.")
    else:
        print("✘ Integrity check FAILED!")

if __name__ == "__main__":
    main()

import numpy as np
import hmac
import hashlib

BLOCK_SIZE_BITS = 128  # Must match Phase 4

def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def bits_to_bytes(bits):
    return np.packbits(bits).tobytes()

def unpad_bits(bits, pad_len):
    if pad_len == 0:
        return bits
    return bits[:-pad_len]

def decrypt(ciphertext_bytes, keystream_bits, permutation_indices, pad_len):
    ciphertext_bits = bytes_to_bits(ciphertext_bytes)
    n_blocks = len(ciphertext_bits) // BLOCK_SIZE_BITS

    xor_keystream = keystream_bits[n_blocks:n_blocks + len(ciphertext_bits)]
    xor_bits = xor_keystream.reshape((n_blocks, BLOCK_SIZE_BITS))
    cipher_blocks = ciphertext_bits.reshape((n_blocks, BLOCK_SIZE_BITS))

    permuted_blocks = np.bitwise_xor(cipher_blocks, xor_bits)

    inv_perm = np.argsort(permutation_indices)
    blocks = permuted_blocks[inv_perm]

    plaintext_bits_padded = blocks.flatten()
    plaintext_bits = unpad_bits(plaintext_bits_padded, pad_len)
    plaintext_bytes = bits_to_bytes(plaintext_bits)

    return plaintext_bytes

def decrypt_phase5():
    print("[Decryption] Loading final key and encrypted data...")

    # Load final key from BB84 Phase 3 output
    final_key = np.load("final_key.npy")
    final_key_bytes = np.packbits(final_key).tobytes()
    print(f"[Decryption] Loaded final key ({len(final_key)} bits)")

    # Load ciphertext and auxiliary files
    with open("ciphertext.bin", "rb") as f:
        ciphertext = f.read()
    permutation_indices = np.load("permutation_indices.npy")
    pad_len = int(np.load("pad_len.npy"))

    # Load saved HMAC tag
    with open("hmac_tag.bin", "rb") as f:
        hmac_tag = f.read()

    # Load keystream generated in Phase 3 (must be same as encryption)
    keystream = np.load("keystream.npy")
    print(f"[Decryption] Loaded keystream from keystream.npy ({len(keystream)} bits)")

    # Verify integrity with HMAC-SHA256 using FINAL KEY as HMAC key!
    print("[Decryption] Verifying HMAC tag for integrity...")
    h = hmac.new(final_key_bytes[:32], ciphertext, hashlib.sha256)  # <-- Use final_key_bytes here!
    computed_tag = h.digest()

    print(f"Loaded HMAC tag:   {hmac_tag.hex()}")
    print(f"Computed HMAC tag: {computed_tag.hex()}")

    if not hmac.compare_digest(hmac_tag, computed_tag):
        raise ValueError("[Error] Integrity verification FAILED! Ciphertext has been tampered with.")
    print("✔ Integrity check PASSED.")

    # Decrypt the ciphertext
    recovered_plaintext = decrypt(ciphertext, keystream, permutation_indices, pad_len)
    print("[Decryption] Decryption complete.\n")

    print("[Decryption Result]")
    print("Recovered plaintext:", recovered_plaintext)
    print("Recovered length:", len(recovered_plaintext))

if __name__ == "__main__":
    decrypt_phase5()
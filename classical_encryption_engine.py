import numpy as np
import hmac
import hashlib

BLOCK_SIZE_BITS = 128  # Block size in bits (16 bytes)

def bytes_to_bits(b):
    """Convert bytes to numpy uint8 bit array."""
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

def bits_to_bytes(bits):
    """Convert numpy uint8 bit array to bytes."""
    return np.packbits(bits).tobytes()

def pad_bits(bits):
    """Pad bit array so its length is multiple of BLOCK_SIZE_BITS."""
    length = len(bits)
    remainder = length % BLOCK_SIZE_BITS
    if remainder == 0:
        return bits, 0
    pad_len = BLOCK_SIZE_BITS - remainder
    padded_bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    return padded_bits, pad_len

def unpad_bits(bits, pad_len):
    """Remove padding bits from bit array."""
    if pad_len == 0:
        return bits
    return bits[:-pad_len]

def encrypt(plaintext_bytes, keystream_bits):
    """
    Encrypt plaintext bytes using keystream bits:
    1. Pad bits to block size
    2. Generate permutation indices
    3. Permute and XOR with keystream
    Returns ciphertext, permutation indices, pad_len
    """
    plaintext_bits = bytes_to_bits(plaintext_bytes)
    padded_bits, pad_len = pad_bits(plaintext_bits)

    n_blocks = len(padded_bits) // BLOCK_SIZE_BITS

    # Generate permutation indices
    permutation_keystream = keystream_bits[:n_blocks]
    permutation_indices = np.argsort(permutation_keystream)

    # XOR keystream
    xor_keystream = keystream_bits[n_blocks:n_blocks + len(padded_bits)]

    blocks = padded_bits.reshape((n_blocks, BLOCK_SIZE_BITS))
    xor_bits = xor_keystream.reshape((n_blocks, BLOCK_SIZE_BITS))

    permuted_blocks = blocks[permutation_indices]
    cipher_blocks = np.bitwise_xor(permuted_blocks, xor_bits)

    ciphertext_bits = cipher_blocks.flatten()
    ciphertext_bytes = bits_to_bytes(ciphertext_bits)

    return ciphertext_bytes, permutation_indices, pad_len

def decrypt(ciphertext_bytes, keystream_bits, permutation_indices, pad_len):
    """
    Decrypt ciphertext using keystream and saved permutation info.
    """
    ciphertext_bits = bytes_to_bits(ciphertext_bytes)
    n_blocks = len(ciphertext_bits) // BLOCK_SIZE_BITS

    xor_keystream = keystream_bits[n_blocks:n_blocks + len(ciphertext_bits)]
    xor_bits = xor_keystream.reshape((n_blocks, BLOCK_SIZE_BITS))
    cipher_blocks = ciphertext_bits.reshape((n_blocks, BLOCK_SIZE_BITS))

    # XOR back
    permuted_blocks = np.bitwise_xor(cipher_blocks, xor_bits)

    # Reverse permutation
    inv_perm = np.argsort(permutation_indices)
    blocks = permuted_blocks[inv_perm]

    plaintext_bits_padded = blocks.flatten()
    plaintext_bits = unpad_bits(plaintext_bits_padded, pad_len)
    plaintext_bytes = bits_to_bytes(plaintext_bits)

    return plaintext_bytes

def test_avalanche_effect(plaintext_bytes, keystream_bits):
    """
    Test avalanche effect by flipping one bit in plaintext.
    """
    original_ciphertext, _, _ = encrypt(plaintext_bytes, keystream_bits)

    flipped_bits = bytes_to_bits(plaintext_bytes).copy()
    flipped_bits[0] ^= 1
    flipped_plaintext = bits_to_bytes(flipped_bits)

    flipped_ciphertext, _, _ = encrypt(flipped_plaintext, keystream_bits)

    ct_bits1 = bytes_to_bits(original_ciphertext)
    ct_bits2 = bytes_to_bits(flipped_ciphertext)

    diff = np.sum(ct_bits1 != ct_bits2)
    ratio = diff / len(ct_bits1) * 100
    print(f"Avalanche effect: {ratio:.2f}% bits changed")
    return ratio

if __name__ == "__main__":
    # === Load keystream from Phase 3 ===
    keystream_bits = np.load("keystream.npy")
    
    # === Define plaintext ===
    plaintext = b"Hello, this is a test message for Phase 4 encryption."

    # === Calculate required keystream length ===
    plaintext_bits_len = len(plaintext) * 8
    padded_len = ((plaintext_bits_len + BLOCK_SIZE_BITS - 1) // BLOCK_SIZE_BITS) * BLOCK_SIZE_BITS
    n_blocks = padded_len // BLOCK_SIZE_BITS
    required_keystream_len = n_blocks + padded_len

    # === Trim or check keystream ===
    if len(keystream_bits) < required_keystream_len:
        raise ValueError("Keystream too short for encryption.")
    keystream_bits = keystream_bits[:required_keystream_len]

    # === Encrypt ===
    ciphertext_bytes, permutation_indices, pad_len = encrypt(plaintext, keystream_bits)

    # === Save Phase 4 outputs ===
    with open("ciphertext.bin", "wb") as f:
        f.write(ciphertext_bytes)
    np.save("permutation_indices.npy", permutation_indices)
    np.save("pad_len.npy", np.array([pad_len]))

    print("✔ Saved: ciphertext.bin, permutation_indices.npy, pad_len.npy")

    # === Generate and save HMAC for Phase 5 ===
    final_key = np.load("final_key.npy")
    final_key_bytes = np.packbits(final_key).tobytes()
    tag = hmac.new(final_key_bytes[:32], ciphertext_bytes, hashlib.sha256).digest()
    with open("hmac_tag.bin", "wb") as f:
        f.write(tag)
    print("✔ HMAC tag saved to hmac_tag.bin")

    # === Decrypt to verify ===
    recovered_plaintext = decrypt(ciphertext_bytes, keystream_bits, permutation_indices, pad_len)

    print("\n=== Verification ===")
    print("Ciphertext (hex):", ciphertext_bytes.hex())
    print("Recovered plaintext:", recovered_plaintext.decode(errors='ignore'))
    print("Original length:", len(plaintext))
    print("Recovered length:", len(recovered_plaintext))

    assert recovered_plaintext == plaintext, "Decryption failed!"
    print("✔ Decryption verified successfully.")

    # === Avalanche Test ===
    test_avalanche_effect(plaintext, keystream_bits)

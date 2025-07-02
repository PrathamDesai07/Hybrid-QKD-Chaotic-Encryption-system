import numpy as np
import hmac
import hashlib
from bb84_pipeline import bb84_pipeline
from ml_map_selection import generate_keystream  # Your Phase 6A function
from classical_encryption_engine import encrypt, decrypt

def test_pipeline(plaintext, qber, iterations=3):
    print(f"\n=== Testing pipeline for QBER={qber:.3f}, plaintext length={len(plaintext)} bytes ===")
    for i in range(iterations):
        print(f"\n-- Iteration {i+1} --")
        
        # Phase 2: Run BB84 pipeline (generate final_key)
        bb84_pipeline()
        final_key = np.load("final_key.npy")
        
        # Phase 3/6A: Generate chaotic keystream using ML-driven map selection
        plaintext_bits_len = len(plaintext) * 8
        block_size_bits = 128
        padded_len = ((plaintext_bits_len + block_size_bits - 1) // block_size_bits) * block_size_bits
        n_blocks = padded_len // block_size_bits
        keystream_len = n_blocks + padded_len
        
        keystream, selected_map = generate_keystream(final_key, qber, keystream_len)
        print(f"Selected chaotic map: {selected_map}")
        print(f"Generated keystream length: {len(keystream)} bits")
        
        # Phase 4: Encrypt
        ciphertext_bytes, permutation_indices, pad_len = encrypt(plaintext, keystream)
        print(f"Ciphertext (hex): {ciphertext_bytes.hex()[:64]}...")  # Truncated for readability
        
        # Phase 4: Decrypt
        recovered_plaintext = decrypt(ciphertext_bytes, keystream, permutation_indices, pad_len)
        
        # Phase 5: HMAC Integrity Check
        final_key_bytes = np.packbits(final_key).tobytes()
        hmac_key = final_key_bytes[:32]
        hmac_tag = hmac.new(hmac_key, ciphertext_bytes, hashlib.sha256).digest()
        computed_tag = hmac.new(hmac_key, ciphertext_bytes, hashlib.sha256).digest()
        integrity_ok = hmac.compare_digest(hmac_tag, computed_tag)
        
        # Validation
        decrypted_ok = recovered_plaintext == plaintext
        print(f"Decryption success: {decrypted_ok}")
        print(f"Integrity check passed: {integrity_ok}")
        
        if not decrypted_ok or not integrity_ok:
            print("ERROR: Pipeline failed integrity or decryption test!")
            break
    print("\n=== Test run completed ===")

if __name__ == "__main__":
    # Define test QBERs and plaintexts (add more if you like)
    qber_values = [0.01, 0.03, 0.05, 0.07, 0.10]
    plaintexts = [
        b"Short message test.",
        b"Hello, this is a test message to verify the full pipeline!",
        b"This is a longer test message to verify that encryption, decryption, and integrity checks work with larger inputs." * 5
    ]
    
    for qber in qber_values:
        for pt in plaintexts:
            test_pipeline(pt, qber, iterations=3)

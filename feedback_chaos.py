import numpy as np
from mpmath import mp
from scipy.stats import chisquare

mp.dps = 50  # set precision


class FeedbackChaosKeystream:
    def __init__(self, final_key_bits, bit_width=32, block_size=10, bits_per_iter=4):
        self.final_key_bits = final_key_bits
        self.bit_width = bit_width
        self.block_size = block_size
        self.bits_per_iter = bits_per_iter
        self.seed_a, self.seed_b = self._split_final_key(final_key_bits)
        self.mu_base = None
        self.x = None
        self.seed_b_pos = 0

    def _split_final_key(self, bits):
        half = len(bits) // 2
        return bits[:half], bits[half:]

    def _bits_to_float(self, bits):
        integer = 0
        for bit in bits:
            integer = (integer << 1) | bit
        max_int = 2 ** self.bit_width
        return mp.mpf(str(integer)) / mp.mpf(str(max_int))

    def _logistic_map(self, x, mu):
        return mu * x * (1 - x)

    def _perturb_parameter(self, base_param, perturb_bits):
        base_int = int(mp.floor(base_param * (2**self.bit_width)))
        perturb_int = 0
        for bit in perturb_bits:
            perturb_int = (perturb_int << 1) | bit
        new_int = base_int ^ perturb_int
        new_param = mp.mpf(str(new_int)) / mp.mpf(str(2**self.bit_width))
        return new_param

    def _circular_bit_rotate(self, val, r):
        val_int = int(mp.floor(val * (2**self.bit_width))) & ((1 << self.bit_width) - 1)
        rotated = ((val_int << r) | (val_int >> (self.bit_width - r))) & ((1 << self.bit_width) - 1)
        return mp.mpf(rotated) / mp.mpf(2**self.bit_width)

    def _nonlinear_mix(self, state_val, perturbed_val, rotate_bits=5):
        rotated = self._circular_bit_rotate(state_val, rotate_bits)
        mixed_int = (int(mp.floor(rotated * (2**self.bit_width))) + int(mp.floor(perturbed_val * (2**self.bit_width)))) % (2**self.bit_width)
        return mp.mpf(mixed_int) / mp.mpf(2**self.bit_width)

    def _quantize_to_bits(self, val):
        scaled = int(mp.floor(val * (2 ** self.bits_per_iter)))
        bits = [(scaled >> i) & 1 for i in reversed(range(self.bits_per_iter))]
        return bits

    def generate_keystream(self, length):
        # Initialize mu_base from seed_a
        self.mu_base = self._bits_to_float(self.seed_a[:self.bit_width]) * mp.mpf('3.99')

        # Initialize logistic map state x
        if len(self.seed_a) > 2 * self.bit_width:
            self.x = self._bits_to_float(self.seed_a[self.bit_width:2*self.bit_width])
        else:
            self.x = mp.mpf('0.5')

        keystream = []
        total_blocks = length // (self.block_size * self.bits_per_iter) + 1

        for _ in range(total_blocks):
            block_bits = []
            for _ in range(self.block_size):
                self.x = self._logistic_map(self.x, self.mu_base)
                block_bits.extend(self._quantize_to_bits(self.x))
            keystream.extend(block_bits)

            if len(keystream) >= length:
                break

            # Perturb mu_base using next bits from seed_b
            if self.seed_b_pos + self.block_size <= len(self.seed_b):
                perturb_bits = self.seed_b[self.seed_b_pos:self.seed_b_pos + self.block_size]
            else:
                end_part = self.seed_b_pos + self.block_size - len(self.seed_b)
                perturb_bits = np.concatenate((self.seed_b[self.seed_b_pos:], self.seed_b[:end_part]))
            self.seed_b_pos = (self.seed_b_pos + self.block_size) % len(self.seed_b)

            self.mu_base = self._perturb_parameter(self.mu_base / mp.mpf('3.99'), perturb_bits) * mp.mpf('3.99')
            self.mu_base = self._nonlinear_mix(self.mu_base / mp.mpf('3.99'), self.mu_base / mp.mpf('3.99')) * mp.mpf('3.99')

        return np.array(keystream[:length], dtype=np.uint8)


def chi_square_test(bits):
    counts = np.bincount(bits, minlength=2)
    expected = np.full(2, len(bits) / 2)
    stat, p = chisquare(counts, expected)
    return stat, p


def test_variability(final_key_bits, runs=5, keystream_len=512):
    print("Testing variability over runs with slight Seed B changes:")
    seed_a = final_key_bits[:len(final_key_bits)//2]
    seed_b_original = final_key_bits[len(final_key_bits)//2:]

    for i in range(runs):
        # Slightly modify seed_b: flip one bit
        seed_b = seed_b_original.copy()
        idx_to_flip = i % len(seed_b)
        seed_b[idx_to_flip] = 1 - seed_b[idx_to_flip]

        combined_seed = np.concatenate((seed_a, seed_b))
        chaos = FeedbackChaosKeystream(combined_seed)
        keystream = chaos.generate_keystream(keystream_len)
        print(f"Run {i+1} first 64 bits: {keystream[:64].tolist()}")


if __name__ == "__main__":
    # Simulate final key bits (512 bits)
    final_key_bits = np.random.randint(0, 2, 512)
    keystream_len = 512

    print("=== Feedback-Driven Quantum-Chaos Key Perturbation Demo with Whitening ===")
    chaos = FeedbackChaosKeystream(final_key_bits)
    keystream = chaos.generate_keystream(keystream_len)

    print(f"Seed A length: {len(chaos.seed_a)} bits, Seed B length: {len(chaos.seed_b)} bits")
    print(f"Generated keystream length: {len(keystream)} bits")
    print(f"First 64 bits: {keystream[:64].tolist()}")

    # Chi-square test for uniformity
    stat, p = chi_square_test(keystream)
    print(f"Chi-square test: statistic={stat:.2f}, p-value={p:.4f}")
    if p < 0.05:
        print("Keystream bits may not be uniform (reject H0).")
    else:
        print("Keystream bits appear uniform (fail to reject H0).")

    # Variability test
    test_variability(final_key_bits, runs=5, keystream_len=keystream_len)

#!/usr/bin/env python3
"""
Generate fuzz corpus for binary quantization.

Creates ≥100 seed samples for the fuzz_quantization target.
Each sample is 3072 bytes (768 f32 values in little-endian format).

Seeds cover:
- All zeros
- All positive
- All negative
- Alternating patterns
- Random distributions
- Edge cases (NaN, Inf, subnormal)
"""

import os
import random
import struct
import sys

CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fuzz", "corpus", "fuzz_quantization"
)

DIM = 768  # Binary quantization dimension
BYTES_PER_FLOAT = 4
SEED_SIZE = DIM * BYTES_PER_FLOAT  # 3072 bytes


def f32_to_bytes(values: list[float]) -> bytes:
    """Convert list of floats to little-endian bytes."""
    return struct.pack(f"<{len(values)}f", *values)


def generate_seeds() -> dict[str, bytes]:
    """Generate all seed samples."""
    seeds = {}

    # === Constant vectors ===
    seeds["all_zeros"] = f32_to_bytes([0.0] * DIM)
    seeds["all_ones"] = f32_to_bytes([1.0] * DIM)
    seeds["all_neg_ones"] = f32_to_bytes([-1.0] * DIM)
    seeds["all_tiny_pos"] = f32_to_bytes([1e-38] * DIM)
    seeds["all_tiny_neg"] = f32_to_bytes([-1e-38] * DIM)
    seeds["all_large_pos"] = f32_to_bytes([1e38] * DIM)
    seeds["all_large_neg"] = f32_to_bytes([-1e38] * DIM)

    # === Alternating patterns ===
    seeds["alt_pos_neg"] = f32_to_bytes(
        [1.0 if i % 2 == 0 else -1.0 for i in range(DIM)]
    )
    seeds["alt_neg_pos"] = f32_to_bytes(
        [-1.0 if i % 2 == 0 else 1.0 for i in range(DIM)]
    )
    seeds["alt_pos_zero"] = f32_to_bytes(
        [1.0 if i % 2 == 0 else 0.0 for i in range(DIM)]
    )
    seeds["alt_neg_zero"] = f32_to_bytes(
        [-1.0 if i % 2 == 0 else 0.0 for i in range(DIM)]
    )
    seeds["alt_4_pattern"] = f32_to_bytes(
        [1.0 if i % 4 < 2 else -1.0 for i in range(DIM)]
    )
    seeds["alt_8_pattern"] = f32_to_bytes(
        [1.0 if i % 8 < 4 else -1.0 for i in range(DIM)]
    )

    # === Special float values ===
    # NaN
    nan_bytes = struct.pack("<f", float("nan"))
    seeds["all_nan"] = nan_bytes * DIM
    seeds["first_nan"] = nan_bytes + f32_to_bytes([1.0] * (DIM - 1))
    seeds["last_nan"] = f32_to_bytes([1.0] * (DIM - 1)) + nan_bytes

    # Infinity
    inf_bytes = struct.pack("<f", float("inf"))
    neg_inf_bytes = struct.pack("<f", float("-inf"))
    seeds["all_inf"] = inf_bytes * DIM
    seeds["all_neg_inf"] = neg_inf_bytes * DIM
    seeds["mixed_inf"] = (inf_bytes + neg_inf_bytes) * (DIM // 2)

    # Negative zero
    neg_zero_bytes = struct.pack("<f", -0.0)
    seeds["all_neg_zero"] = neg_zero_bytes * DIM
    seeds["mixed_zero_neg_zero"] = (struct.pack("<f", 0.0) + neg_zero_bytes) * (
        DIM // 2
    )

    # Subnormal (denormalized) numbers
    subnormal = 1e-45  # Smallest positive subnormal f32
    seeds["all_subnormal"] = f32_to_bytes([subnormal] * DIM)
    seeds["all_neg_subnormal"] = f32_to_bytes([-subnormal] * DIM)

    # === Random distributions ===
    random.seed(42)  # Reproducible

    # Uniform distributions
    for i in range(20):
        seeds[f"random_uniform_{i:02d}"] = f32_to_bytes(
            [random.uniform(-1.0, 1.0) for _ in range(DIM)]
        )

    # Normal distribution
    for i in range(10):
        seeds[f"random_normal_{i:02d}"] = f32_to_bytes(
            [random.gauss(0.0, 1.0) for _ in range(DIM)]
        )

    # Sparse vectors (mostly zeros)
    for i in range(10):
        sparsity = 0.9 + i * 0.01  # 90% to 99% zeros
        seeds[f"sparse_{int(sparsity * 100):02d}pct"] = f32_to_bytes(
            [
                random.uniform(-1.0, 1.0) if random.random() > sparsity else 0.0
                for _ in range(DIM)
            ]
        )

    # Clustered values (similar to real embeddings)
    for i in range(10):
        center = random.uniform(-0.5, 0.5)
        spread = 0.1 + i * 0.02
        seeds[f"clustered_{i:02d}"] = f32_to_bytes(
            [center + random.gauss(0.0, spread) for _ in range(DIM)]
        )

    # === Boundary patterns ===
    # Single bit patterns
    for bit_pos in [0, 7, 8, 15, 16, 383, 384, 767]:
        vec = [0.0] * DIM
        vec[bit_pos] = 1.0
        seeds[f"single_bit_{bit_pos:03d}"] = f32_to_bytes(vec)

    # Byte-aligned patterns
    for byte_idx in [0, 1, 47, 48, 95]:
        vec = [0.0] * DIM
        start = byte_idx * 8
        for j in range(8):
            if start + j < DIM:
                vec[start + j] = 1.0
        seeds[f"byte_aligned_{byte_idx:02d}"] = f32_to_bytes(vec)

    # Gradient vectors
    seeds["gradient_linear"] = f32_to_bytes(
        [
            (i / DIM) * 2 - 1
            for i in range(DIM)  # -1 to +1
        ]
    )
    seeds["gradient_reverse"] = f32_to_bytes(
        [
            1 - (i / DIM) * 2
            for i in range(DIM)  # +1 to -1
        ]
    )

    # Step functions
    for step_pos in [DIM // 4, DIM // 2, 3 * DIM // 4]:
        seeds[f"step_at_{step_pos:03d}"] = f32_to_bytes(
            [-1.0 if i < step_pos else 1.0 for i in range(DIM)]
        )

    # === Malformed data (may cause panics if not handled) ===
    # Very short data
    seeds["short_1_byte"] = b"\x00"
    seeds["short_100_bytes"] = b"\x00" * 100
    seeds["short_3071_bytes"] = b"\x00" * 3071  # One byte short

    # Very long data
    seeds["long_4096_bytes"] = b"\x00" * 4096
    seeds["long_8192_bytes"] = b"\x00" * 8192

    # Random bytes (not valid f32 patterns)
    seeds["random_bytes"] = bytes(random.randint(0, 255) for _ in range(SEED_SIZE))

    # === Additional patterns to reach 100 ===
    # Power of 2 boundaries
    seeds["half_pos_half_neg"] = f32_to_bytes([1.0] * (DIM // 2) + [-1.0] * (DIM // 2))
    seeds["thirds_pattern"] = f32_to_bytes(
        [
            1.0 if i < DIM // 3 else (-1.0 if i < 2 * DIM // 3 else 0.0)
            for i in range(DIM)
        ]
    )
    seeds["quarters_pattern"] = f32_to_bytes(
        [
            1.0
            if i % 4 == 0
            else (-1.0 if i % 4 == 1 else (0.5 if i % 4 == 2 else -0.5))
            for i in range(DIM)
        ]
    )

    # Very small positive/negative values (near decision boundary)
    seeds["tiny_epsilon"] = f32_to_bytes([1e-7] * DIM)
    seeds["tiny_neg_epsilon"] = f32_to_bytes([-1e-7] * DIM)
    seeds["alternating_epsilon"] = f32_to_bytes(
        [1e-7 if i % 2 == 0 else -1e-7 for i in range(DIM)]
    )

    return seeds


def main():
    """Generate and write all corpus seeds."""
    os.makedirs(CORPUS_DIR, exist_ok=True)

    seeds = generate_seeds()

    for name, data in seeds.items():
        path = os.path.join(CORPUS_DIR, name)
        with open(path, "wb") as f:
            f.write(data)
        print(f"  Created: {name} ({len(data)} bytes)")

    print(f"\nGenerated {len(seeds)} seed files in {CORPUS_DIR}")

    # Verify we have at least 100 seeds
    if len(seeds) < 100:
        print(f"WARNING: Only {len(seeds)} seeds, need at least 100!")
        return 1

    print("SUCCESS: Corpus generation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

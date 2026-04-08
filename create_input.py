#!/usr/bin/env python3
"""
Create DCTQ input JSON for circuit processing.

Reshapes HD image (1280×720) to 160×5760 format:
- Original: 1280 width × 720 height
- Reshaped: 160 width × 5760 height
- Process: 8 rows at a time = 5760/8 = 720 steps

Each step processes 8×160 blocks (20 blocks of 8×8 each)
"""

import numpy as np
from PIL import Image
import json
import sys
from dctq import JPEG_QUANT_LUMINANCE, JPEG_QUANT_CHROMINANCE
from dctq_approx import apply_dctq_approx


def reshape_image_1280x720_to_160x5760(image):
    """
    Reshape HD image from 1280×720 to 160×5760.

    Strategy: Treat the wide image as a tall image
    - Take 8 consecutive horizontal blocks (8×8 each = 64 pixels wide)
    - Stack them vertically

    Args:
        image: np.array of shape (720, 1280, 3)

    Returns:
        np.array of shape (5760, 160, 3)
    """
    height, width, channels = image.shape
    assert height == 720 and width == 1280, f"Expected 1280×720, got {width}×{height}"

    # Original: 1280 pixels wide = 160 blocks of 8 pixels
    # We want to reshape to 160 pixels wide

    # New dimensions
    new_width = 160  # 160 pixels = 20 blocks of 8
    new_height = 5760  # 720 rows × 8 (since we're expanding width into height)

    output = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    # Process each original row
    for orig_row in range(height):
        # Each original row of 1280 pixels becomes 8 rows of 160 pixels
        # Split 1280 pixels into 8 chunks of 160 pixels each
        for chunk_idx in range(8):
            new_row = orig_row * 8 + chunk_idx
            start_col = chunk_idx * 160
            end_col = start_col + 160

            output[new_row, :, :] = image[orig_row, start_col:end_col, :]

    return output


def apply_dctq_transformation(image):
    """
    Apply DCTQ transformation using approximated DCT (matches circuit).

    Args:
        image: np.array (height, width, 3)

    Returns:
        transformed: np.array (height, width, 3) - quantized DCT coefficients
    """
    # Quantization tables for each channel
    quant_tables = [
        JPEG_QUANT_LUMINANCE,      # Red/Luminance
        JPEG_QUANT_CHROMINANCE,    # Green/Chrominance
        JPEG_QUANT_CHROMINANCE     # Blue/Chrominance
    ]

    # Use approximated DCTQ that matches the circuit exactly
    return apply_dctq_approx(image, quant_tables)


def compress_pixel(r, g, b):
    """
    Compress RGB pixel into single field element (same as existing compression).
    Format: r * 2^16 + g * 2^8 + b

    Values are clamped to 0-255 range for standard pixel data.
    """
    r_val = int(max(0, min(255, r)))
    g_val = int(max(0, min(255, g)))
    b_val = int(max(0, min(255, b)))

    return r_val * 65536 + g_val * 256 + b_val


def create_compressed_rows(image):
    """
    Create compressed row representation.
    Each 10 pixels compressed into one field element.

    Args:
        image: (height, width, 3) where width must be divisible by 10

    Returns:
        List of compressed rows
    """
    height, width, channels = image.shape
    assert width % 10 == 0, f"Width {width} must be divisible by 10"

    compressed_width = width // 10
    compressed_rows = []

    for row_idx in range(height):
        compressed_row = []
        for i in range(0, width, 10):
            # Take 10 pixels and compress ALL of them into one element
            pixels = image[row_idx, i:i+10, :]  # Shape: (10, 3)

            # Pack all 10 RGB pixels into a single 240-bit integer
            # Format: pixel_i channel_j goes at bit position: i*24 + j*8
            compressed = 0
            for pixel_idx in range(10):
                r, g, b = pixels[pixel_idx]
                compressed |= (int(r) << (pixel_idx * 24 + 0))
                compressed |= (int(g) << (pixel_idx * 24 + 8))
                compressed |= (int(b) << (pixel_idx * 24 + 16))

            compressed_row.append(f"0x{compressed:x}")

        compressed_rows.append(compressed_row)

    return compressed_rows


def create_dctq_input_json(input_image_path, output_json_path):
    """
    Create DCTQ input JSON file.

    Process:
    1. Load HD image (1280×720)
    2. Reshape to 160×5760
    3. Apply DCTQ transformation
    4. Create compressed format
    5. Generate JSON with 8-row batches (720 steps)
    """
    print("=" * 70)
    print("Creating DCTQ Input JSON")
    print("=" * 70)
    print()

    # Load image
    print(f"Loading image: {input_image_path}")
    img = Image.open(input_image_path).convert("RGB")
    img = img.resize((1280, 720))
    img_array = np.array(img)
    print(f"Original shape: {img_array.shape}")
    print()

    # Reshape to 160×5760
    print("Reshaping 1280×720 → 160×5760...")
    reshaped = reshape_image_1280x720_to_160x5760(img_array)
    print(f"Reshaped shape: {reshaped.shape}")
    print()

    # Apply DCTQ transformation
    print("Applying DCTQ transformation (DCT + Quantization)...")
    transformed = apply_dctq_transformation(reshaped)
    print("Transformation complete!")
    print()

    # Compress both images to VIMz format
    print("Compressing to VIMz circuit format...")
    compressed_orig = create_compressed_rows(reshaped)

    # For transformed (DCTQ coefficients), we need to handle them carefully
    # DCTQ coefficients can be negative and large, but we'll clamp them to a reasonable range
    # that the circuit can handle
    transformed_clamped = np.clip(transformed, -127, 127) + 128  # Shift to 0-255 range
    compressed_tran = create_compressed_rows(transformed_clamped.astype(np.uint8))
    print(f"Compressed format: {len(compressed_orig)} rows × {len(compressed_orig[0])} compressed elements")
    print()

    # Create JSON structure compatible with VIMz
    print("Creating JSON structure...")
    print(f"Rows: {len(compressed_orig)}")
    print(f"Compressed width: {len(compressed_orig[0])}")
    print()

    json_data = {
        "original": compressed_orig,
        "transformed": compressed_tran
    }

    # Write JSON
    print(f"Writing to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print()
    print("=" * 70)
    print("Success!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Input: {input_image_path}")
    print(f"  Output: {output_json_path}")
    print(f"  Original: 1280×720 ({img_array.shape})")
    print(f"  Reshaped: 160×5760 ({reshaped.shape})")
    print(f"  Steps: 720")
    print(f"  Blocks per step: 20 (8×8 blocks)")
    print()

    # Verification
    print("Verification:")
    print(f"  Original min/max: {reshaped.min()}, {reshaped.max()}")
    print(f"  Transformed min/max: {transformed.min()}, {transformed.max()}")
    print(f"  Non-zero coefficients: {np.count_nonzero(transformed)}")
    print(f"  Zero coefficients: {np.count_nonzero(transformed == 0)}")
    sparsity = (np.count_nonzero(transformed == 0) / transformed.size) * 100
    print(f"  Sparsity: {sparsity:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_dctq_input.py <input_image> <output_json>")
        print()
        print("Example:")
        print("  python create_dctq_input.py samples/images/test.jpg dctq_input.json")
        print()
        print("This script:")
        print("  1. Loads HD image (1280×720)")
        print("  2. Reshapes to 160×5760")
        print("  3. Applies DCTQ transformation")
        print("  4. Generates JSON for circuit input")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    create_dctq_input_json(input_path, output_path)
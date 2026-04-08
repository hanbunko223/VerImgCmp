#!/usr/bin/env python3
"""
Create raw-pixel input JSON for callnova.

The input image is:
- loaded as RGB
- It is supposed to be HD. If not, it will be resized to HD (1280x720)
- reshaped to 160x5760

Output JSON format:
{
  "original": [
    [[r, g, b], ... 160 pixels ...],
    ... 5760 rows ...
  ]
}
"""

import json
import sys

import numpy as np
from PIL import Image


HD_WIDTH = 1280
HD_HEIGHT = 720
RESHAPED_WIDTH = 160
RESHAPED_HEIGHT = 5760
HORIZONTAL_CHUNKS = HD_WIDTH // RESHAPED_WIDTH


def reshape_image_1280x720_to_160x5760(image):
    height, width, channels = image.shape
    assert height == HD_HEIGHT and width == HD_WIDTH, (
        f"Expected {HD_WIDTH}x{HD_HEIGHT}, got {width}x{height}"
    )

    output = np.zeros((RESHAPED_HEIGHT, RESHAPED_WIDTH, channels), dtype=image.dtype)

    for orig_row in range(height):
        for chunk_idx in range(HORIZONTAL_CHUNKS):
            new_row = orig_row * HORIZONTAL_CHUNKS + chunk_idx
            start_col = chunk_idx * RESHAPED_WIDTH
            end_col = start_col + RESHAPED_WIDTH
            output[new_row, :, :] = image[orig_row, start_col:end_col, :]

    return output


def create_input_json(input_image_path, output_json_path):
    image = Image.open(input_image_path).convert("RGB")
    image = image.resize((HD_WIDTH, HD_HEIGHT))
    image_array = np.array(image, dtype=np.uint8)
    reshaped = reshape_image_1280x720_to_160x5760(image_array)

    json_data = {
        "original": reshaped.tolist(),
    }

    with open(output_json_path, "w") as output_file:
        json.dump(json_data, output_file, indent=2)

    print(f"Wrote raw-pixel input to {output_json_path}")
    print(f"Rows: {len(json_data['original'])}")
    print(f"Pixels per row: {len(json_data['original'][0])}")
    print(f"Channels per pixel: {len(json_data['original'][0][0])}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_input.py <input_image> <output_json>")
        sys.exit(1)

    create_input_json(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""
Create raw-pixel input JSON for callnova_hash.

The step circuit always consumes 64 packed rows x 160 pixels, i.e. 10240 pixels
per step. Images are encoded as:
1. load/resize to the requested resolution,
2. flatten pixels in raster order,
3. reshape into logical rows of width 1280,
4. zero-pad trailing pixels to a multiple of 8 logical rows,
5. split each 1280-wide logical row into 8 packed rows of width 160.

The output wire shape remains:
{
  "original": [
    [[r, g, b], ... 160 pixels ...],
    ... packed rows ...
  ]
}
"""

import argparse
import json
import math
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
LOGICAL_WIDTH = 1280
PACKED_ROW_WIDTH = 160
PACKED_ROWS_PER_STEP = 64
CHUNKS_PER_LOGICAL_ROW = LOGICAL_WIDTH // PACKED_ROW_WIDTH
LOGICAL_ROWS_PER_STEP = PACKED_ROWS_PER_STEP // CHUNKS_PER_LOGICAL_ROW

RESOLUTION_DIMS = {
    "SD": (640, 480),
    "HD": (1280, 720),
    "FHD": (1920, 1080),
    "QHD": (2560, 1440),
    "4K": (3840, 2160),
}

SAMPLE_CANDIDATES = {
    "SD": ["SD.png", "4K.png"],
    "HD": ["HD.png", "4K.png"],
    "FHD": ["FHD.png", "4K.png"],
    "QHD": ["QHD.png", "2K.png", "4K.png"],
    "4K": ["4K.png"],
}


@dataclass(frozen=True)
class EncodingPlan:
    resolution: str
    width: int
    height: int
    logical_rows: int
    padded_logical_rows: int
    packed_rows: int
    step_count: int
    padded_pixels: int


def resize_filter():
    return getattr(Image, "Resampling", Image).LANCZOS


def plan_for_dimensions(width, height, resolution):
    pixel_count = width * height
    logical_rows = math.ceil(pixel_count / LOGICAL_WIDTH)
    padded_logical_rows = math.ceil(logical_rows / LOGICAL_ROWS_PER_STEP) * LOGICAL_ROWS_PER_STEP
    padded_pixels = padded_logical_rows * LOGICAL_WIDTH - pixel_count
    packed_rows = padded_logical_rows * CHUNKS_PER_LOGICAL_ROW
    step_count = packed_rows // PACKED_ROWS_PER_STEP
    return EncodingPlan(
        resolution=resolution,
        width=width,
        height=height,
        logical_rows=logical_rows,
        padded_logical_rows=padded_logical_rows,
        packed_rows=packed_rows,
        step_count=step_count,
        padded_pixels=padded_pixels,
    )


def resolve_sample_image(resolution):
    sample_dir = REPO_ROOT / "samples"
    for candidate in SAMPLE_CANDIDATES[resolution]:
        candidate_path = sample_dir / candidate
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(f"no sample image found for {resolution}")


def load_resolution_image(resolution, input_image_path=None):
    target_width, target_height = RESOLUTION_DIMS[resolution]
    source_path = Path(input_image_path) if input_image_path else resolve_sample_image(resolution)
    img = Image.open(source_path).convert("RGB")
    resized = img.size != (target_width, target_height)
    if resized:
        img = img.resize((target_width, target_height), resize_filter())
    return source_path, resized, np.array(img, dtype=np.uint8)


def encode_flat_pixels_to_packed_rows(
    flat_pixels,
    logical_width=LOGICAL_WIDTH,
    packed_row_width=PACKED_ROW_WIDTH,
    packed_rows_per_step=PACKED_ROWS_PER_STEP,
):
    chunks_per_logical_row = logical_width // packed_row_width
    logical_rows_per_step = packed_rows_per_step // chunks_per_logical_row
    logical_rows = math.ceil(len(flat_pixels) / logical_width)
    padded_logical_rows = math.ceil(logical_rows / logical_rows_per_step) * logical_rows_per_step
    padded_pixels = padded_logical_rows * logical_width - len(flat_pixels)

    if padded_pixels:
        padding = np.zeros((padded_pixels, flat_pixels.shape[1]), dtype=flat_pixels.dtype)
        flat_pixels = np.concatenate([flat_pixels, padding], axis=0)

    logical = flat_pixels.reshape(padded_logical_rows, logical_width, flat_pixels.shape[1])
    packed = logical.reshape(
        padded_logical_rows, chunks_per_logical_row, packed_row_width, flat_pixels.shape[1]
    ).reshape(padded_logical_rows * chunks_per_logical_row, packed_row_width, flat_pixels.shape[1])

    return packed, {
        "logical_rows": logical_rows,
        "padded_logical_rows": padded_logical_rows,
        "packed_rows": packed.shape[0],
        "step_count": packed.shape[0] // packed_rows_per_step,
        "padded_pixels": padded_pixels,
    }


def encode_image_to_packed_rows(image_array, resolution):
    height, width, channels = image_array.shape
    assert channels == 3, f"Expected RGB image, got {channels} channels"
    plan = plan_for_dimensions(width, height, resolution)
    flat_pixels = image_array.reshape(-1, channels)
    packed_rows, metadata = encode_flat_pixels_to_packed_rows(flat_pixels)
    assert metadata["logical_rows"] == plan.logical_rows
    assert metadata["padded_logical_rows"] == plan.padded_logical_rows
    assert metadata["packed_rows"] == plan.packed_rows
    assert metadata["step_count"] == plan.step_count
    assert metadata["padded_pixels"] == plan.padded_pixels
    return packed_rows, plan


def create_dctq_pixels_input_json(resolution, output_json_path, input_image_path=None):
    source_path, resized, image_array = load_resolution_image(resolution, input_image_path)
    packed_rows, plan = encode_image_to_packed_rows(image_array, resolution)

    json_data = {
        "original": packed_rows.astype(int).tolist(),
    }

    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2)

    print("=" * 70)
    print("Raw-pixel callnova_hash input created")
    print("=" * 70)
    print(f"resolution:           {resolution} ({plan.width}x{plan.height})")
    print(f"input:                {source_path}")
    print(f"output:               {output_json_path}")
    print(f"resized:              {'yes' if resized else 'no'}")
    print(f"logical rows:         {plan.logical_rows}")
    print(f"padded logical rows:  {plan.padded_logical_rows}")
    print(f"packed rows:          {plan.packed_rows}")
    print(f"steps:                {plan.step_count}")
    print(f"padded pixels:        {plan.padded_pixels}")
    print(f"pixels per packed row:{PACKED_ROW_WIDTH}")
    print(f"packed rows per step: {PACKED_ROWS_PER_STEP}")


class GeneratorTests(unittest.TestCase):
    def test_raster_flatten_then_pack_preserves_row_major_order(self):
        image = np.array(
            [
                [[1, 0, 0], [2, 0, 0], [3, 0, 0]],
                [[4, 0, 0], [5, 0, 0], [6, 0, 0]],
            ],
            dtype=np.uint8,
        )
        flat = image.reshape(-1, 3)
        packed_rows, metadata = encode_flat_pixels_to_packed_rows(
            flat,
            logical_width=4,
            packed_row_width=2,
            packed_rows_per_step=4,
        )

        self.assertEqual(metadata["logical_rows"], 2)
        self.assertEqual(metadata["padded_pixels"], 2)
        self.assertEqual(
            packed_rows[:, :, 0].tolist(),
            [[1, 2], [3, 4], [5, 6], [0, 0]],
        )

    def test_fhd_padding_is_exactly_5120_pixels(self):
        plan = plan_for_dimensions(*RESOLUTION_DIMS["FHD"], "FHD")
        self.assertEqual(plan.padded_pixels, 5120)
        self.assertEqual(plan.step_count, 203)

    def test_four_k_uses_810_steps(self):
        plan = plan_for_dimensions(*RESOLUTION_DIMS["4K"], "4K")
        self.assertEqual(plan.step_count, 810)
        self.assertEqual(plan.padded_pixels, 0)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create raw-pixel input JSON for callnova_hash"
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Resolution name (SD/HD/FHD/QHD/4K) or an input image path",
    )
    parser.add_argument("output", nargs="?", help="Output JSON path")
    parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_DIMS),
        help="Resolution to use when `source` is an image path. Defaults to HD for backward compatibility.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal generator tests and exit.",
    )
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeneratorTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if args.source is None or args.output is None:
        print(
            "Usage: python create_dctq_pixels_input.py <SD|HD|FHD|QHD|4K> <output_json>\n"
            "   or: python create_dctq_pixels_input.py <input_image> <output_json> --resolution <SD|HD|FHD|QHD|4K>"
        )
        return 1

    source_upper = args.source.upper()
    if source_upper in RESOLUTION_DIMS:
        resolution = source_upper
        input_image_path = None
    else:
        resolution = args.resolution or "HD"
        input_image_path = args.source

    create_dctq_pixels_input_json(resolution, args.output, input_image_path=input_image_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

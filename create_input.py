#!/usr/bin/env python3
"""Prepare RGB witness pixels; generate coefficients with the Rust CLI."""
import argparse
import json
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parent
SIZES = {'SD': (640, 480), 'HD': (1280, 720), 'FHD': (1920, 1080),
         'QHD': (2560, 1440), '4K': (3840, 2160)}
def pack_pixels(pixels):
    pixels = [list(p) for p in pixels]
    pixels.extend([[0, 0, 0] for _ in range((-len(pixels)) % 2560)])
    return [pixels[i:i + 160] for i in range(0, len(pixels), 160)]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Resolution name or image path')
    parser.add_argument('output', type=Path)
    parser.add_argument('--resolution', choices=SIZES)
    args = parser.parse_args()
    resolution = args.resolution or (args.source if args.source in SIZES else 'HD')
    source = ROOT / 'samples' / (args.source + '.png') if args.source in SIZES else Path(args.source)
    with Image.open(source) as original:
        image = original.convert('RGB')
        if image.size != SIZES[resolution]:
            image = image.resize(SIZES[resolution], Image.Resampling.LANCZOS)
        raw = image.tobytes()
    rows = pack_pixels(zip(raw[0::3], raw[1::3], raw[2::3]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'original': rows}, separators=(',', ':')) + '\n')
    print(json.dumps({'resolution': resolution, 'source': str(source),
                     'output': str(args.output), 'packed_rows': len(rows), 'step_count': len(rows) // 16}))
if __name__ == '__main__':
    main()

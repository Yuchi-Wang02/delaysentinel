#!/usr/bin/env python
"""Export a Hugging Face ``training_args.bin`` to JSON (needs torch).

python scripts/export_training_config.py <training_args.bin> runs/sc904/training_config.json
"""

import json
import sys

import torch


def main(src: str, dst: str) -> None:
    args = torch.load(src, weights_only=False, map_location="cpu")
    payload = args.to_dict()
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=str, sort_keys=True)
    print(f"wrote {dst} ({len(payload)} keys)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

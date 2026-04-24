"""
Utilities for reformatting flat image folders into the GenEval directory structure.

GenEval expects:
  <IMAGE_FOLDER>/
      00000/
          metadata.jsonl   ← N-th line from evaluation_metadata.jsonl
          samples/
              0000.png
              0001.png
              ...
      00001/
          ...

Input flat folder has files named: {image_id:06d}_seed{N}.png
e.g. 000042_seed10.png, 000042_seed20.png, 000042_seed31.png, 000042_seed42.png
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Union


DEFAULT_METADATA = Path(__file__).parent / "geneval" / "prompts" / "evaluation_metadata.jsonl"


def reformat_to_geneval(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    metadata_path: Union[str, Path] = DEFAULT_METADATA,
) -> None:
    """Reorganize a flat folder of {image_id}_seed{N}.png images into GenEval format.

    Each unique image_id becomes a subfolder named with that id (zero-padded to 5 digits).
    Seeds are sorted and renamed 0000.png, 0001.png, ... inside samples/.
    A metadata.jsonl is written per subfolder from the corresponding line in
    evaluation_metadata.jsonl.

    Args:
        input_dir: Folder containing flat {image_id}_seed{N}.png files.
        output_dir: Destination folder (created if it does not exist).
        metadata_path: Path to GenEval evaluation_metadata.jsonl.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)

    # Load all metadata lines indexed by 0-based prompt index.
    with metadata_path.open() as f:
        metadata_lines = [line.rstrip("\n") for line in f if line.strip()]

    # Group images by image_id.
    pattern = re.compile(r"^(\d+)_seed(\d+)\.png$")
    groups: defaultdict[int, list[tuple[int, Path]]] = defaultdict(list)
    for img_path in sorted(input_dir.glob("*.png")):
        m = pattern.match(img_path.name)
        if not m:
            continue
        image_id, seed = int(m.group(1)), int(m.group(2))
        groups[image_id].append((seed, img_path))

    if not groups:
        raise ValueError(f"No matching images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_id in sorted(groups):
        seeds = sorted(groups[image_id], key=lambda x: x[0])  # sort by seed value

        subfolder = output_dir / f"{image_id:05d}"
        samples_dir = subfolder / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Copy images as 0000.png, 0001.png, ...
        for sample_idx, (_, src_path) in enumerate(seeds):
            dst = samples_dir / f"{sample_idx:04d}.png"
            shutil.copy2(src_path, dst)

        # Write per-prompt metadata.jsonl (single line = N-th entry from master file).
        if image_id < len(metadata_lines):
            meta_line = metadata_lines[image_id]
        else:
            # Fallback: store minimal metadata so evaluation can still run.
            meta_line = json.dumps({"prompt": "", "tag": "unknown"})

        (subfolder / "metadata.jsonl").write_text(meta_line + "\n")

    print(
        f"Reformatted {len(groups)} prompts "
        f"({sum(len(v) for v in groups.values())} images) → {output_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reformat flat image folder to GenEval directory structure."
    )
    parser.add_argument("input_dir", help="Flat folder with {image_id}_seed{N}.png files")
    parser.add_argument("output_dir", help="Output folder in GenEval format")
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="Path to evaluation_metadata.jsonl (default: geneval/prompts/evaluation_metadata.jsonl)",
    )
    args = parser.parse_args()
    reformat_to_geneval(args.input_dir, args.output_dir, args.metadata)

#!/usr/bin/env python3
"""
Unified IQA Evaluation Script

Metrics:
  image_reward   - ImageReward score [prompt-conditioned, higher=better]
  clipiqa        - CLIP-IQA via PyIQA [no-reference, higher=better]
  niqe           - NIQE via PyIQA [no-reference, lower=better]
  t2icompbench   - T2I-CompBench (spatial / non-spatial / complex) [prompt-conditioned]
  geneval        - GenEval object-detection alignment [prompt-conditioned]

Prompt-conditioned metrics require --prompts_file.
T2I-CompBench and GenEval also require their repos to be cloned via setup.sh.

Image directory layouts:
  Flat   — images_dir contains image files directly (1 seed per prompt):
             images_dir/img001.jpg  img002.jpg  ...
  Folders — images_dir contains one subfolder per prompt; images inside are seeds:
             images_dir/00000/0000.jpg 0001.jpg 0002.jpg 0003.jpg
                        00001/0000.jpg 0001.jpg ...
           Per metrics.md, the standard protocol samples each prompt with 4 seeds.

Usage:
  # No-reference only (no prompts needed)
  python evaluate_iqa.py --images_dir /path/to/images

  # Flat layout, 1 image per prompt (e.g. coco5k)
  python evaluate_iqa.py \\
      --images_dir /local/howard/.../coco5k_val_images \\
      --prompts_file /local/howard/.../coco5k_val.txt \\
      --metrics image_reward clipiqa niqe t2icompbench geneval \\
      --output_dir ./results

  # Folder layout, 4 seeds per prompt (standard T2I-CompBench / GenEval protocol)
  python evaluate_iqa.py \\
      --images_dir /path/to/images_by_prompt/ \\
      --prompts_file /path/to/prompts.txt \\
      --metrics image_reward clipiqa niqe t2icompbench geneval

  # GenEval with explicit per-prompt task types
  python evaluate_iqa.py \\
      --images_dir /path/to/images_by_prompt/ \\
      --prompts_file /path/to/prompts.txt \\
      --metrics geneval \\
      --geneval_tasks_file tasks.json
      # tasks.json: list of task strings, one per prompt: ["two_obj", "color", ...]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# GenEval task names used by evaluate_images.py (per metrics.md subset)
GENEVAL_TASKS = ["two_obj", "counting", "color", "color_attribution"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_images_and_prompts(images_dir: str, prompts_file: str | None):
    """
    Auto-detects the directory layout and returns (images, prompts, grouped).

    Flat layout  — images_dir contains image files directly:
        images_dir/img001.jpg, img002.jpg, ...
        → one image per prompt, seeds = 1

    Folder layout — images_dir contains subdirectories:
        images_dir/00000/0000.jpg, 0001.jpg, ...
                   00001/0000.jpg, 0001.jpg, ...
        → each subdirectory is one prompt; images inside are the seeds.
          The number of seeds is the image count in each subfolder.

    Returns:
      images   - flat sorted list of every image Path
      prompts  - list of prompt strings (one per group), or None
      grouped  - list of lists: grouped[i] = [seed_0, seed_1, ...] for prompt i
    """
    root = Path(images_dir)
    children = sorted(root.iterdir())
    subdirs = [c for c in children if c.is_dir()]
    flat_imgs = [c for c in children if c.is_file() and c.suffix.lower() in IMAGE_EXTENSIONS]

    if subdirs and not flat_imgs:
        # Folder-of-folders layout
        grouped = []
        for subdir in subdirs:
            seeds = sorted(
                p for p in subdir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if seeds:
                grouped.append(seeds)
        if not grouped:
            raise FileNotFoundError(f"No images found in subdirectories of {images_dir}")
        images = [img for group in grouped for img in group]
        seeds_per = len(grouped[0])
        print(f"Folder layout detected: {len(grouped)} prompt folder(s), "
              f"{seeds_per} seed(s) each")
    elif flat_imgs:
        # Flat layout — one image per prompt
        images = flat_imgs
        grouped = [[img] for img in images]
        print(f"Flat layout detected: {len(images)} image(s), 1 seed each")
    else:
        raise FileNotFoundError(f"No images or subdirectories found in {images_dir}")

    prompts = None
    if prompts_file:
        with open(prompts_file) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if len(prompts) != len(grouped):
            raise ValueError(
                f"Prompt count ({len(prompts)}) != group count ({len(grouped)}). "
                f"Prompts file must have one line per prompt group."
            )

    return images, prompts, grouped


def _safe_prompt_for_filename(prompt: str, max_len: int = 80) -> str:
    import re as _re
    safe = prompt[:max_len]
    safe = _re.sub(r'[^A-Za-z0-9 ]', '', safe)
    return safe.strip().replace(" ", "_")


# ---------------------------------------------------------------------------
# ImageReward
# ---------------------------------------------------------------------------

def run_image_reward(grouped_images, prompts, device):
    """ImageReward: text-conditioned human-preference score.
    When multiple seeds exist per prompt, scores are averaged per prompt.
    """
    import ImageReward as RM

    print("Loading ImageReward-v1.0 …")
    model = RM.load("ImageReward-v1.0")

    per_prompt = {}
    per_image = {}
    for seed_imgs, prompt in tqdm(zip(grouped_images, prompts), total=len(prompts),
                                  desc="ImageReward"):
        seed_scores = []
        for img_path in tqdm(seed_imgs, desc="  seeds", leave=False):
            s = float(model.score(prompt, str(img_path)))
            per_image[img_path.name] = s
            seed_scores.append(s)
        per_prompt[prompt] = sum(seed_scores) / len(seed_scores)

    mean = sum(per_prompt.values()) / len(per_prompt)
    print(f"  mean ImageReward: {mean:.4f}")
    return {"per_prompt": per_prompt, "per_image": per_image, "mean": mean}


# ---------------------------------------------------------------------------
# CLIP-IQA  (PyIQA, no-reference)
# ---------------------------------------------------------------------------

def run_clipiqa(images, device):
    """CLIP-IQA: no-reference perceptual quality (default prompts)."""
    import pyiqa

    print("Loading clipiqa …")
    metric = pyiqa.create_metric("clipiqa", device=device)

    per_image = {}
    for img_path in tqdm(images, desc="CLIP-IQA"):
        per_image[img_path.name] = float(metric(str(img_path)))

    mean = sum(per_image.values()) / len(per_image)
    print(f"  mean CLIP-IQA: {mean:.4f}")
    return {"per_image": per_image, "mean": mean, "lower_better": False}


# ---------------------------------------------------------------------------
# NIQE  (PyIQA, no-reference)
# ---------------------------------------------------------------------------

def run_niqe(images, device):
    """NIQE: no-reference natural image quality (default settings)."""
    import pyiqa

    print("Loading niqe …")
    metric = pyiqa.create_metric("niqe", device=device)

    per_image = {}
    for img_path in tqdm(images, desc="NIQE"):
        per_image[img_path.name] = float(metric(str(img_path)))

    mean = sum(per_image.values()) / len(per_image)
    print(f"  mean NIQE: {mean:.4f}  (lower is better)")
    return {"per_image": per_image, "mean": mean, "lower_better": True}


# ---------------------------------------------------------------------------
# T2I-CompBench
# ---------------------------------------------------------------------------

def run_t2icompbench(grouped_images, prompts, t2i_root: str, output_dir: Path):
    """
    T2I-CompBench: spatial / non-spatial / complex compositional alignment.

    Category → evaluator mapping (per metrics.md):
      Spatial     → UniDet 2D spatial + 3D spatial
      Non-spatial → BLIP-VQA (attribute binding) + CLIPScore
      Complex     → 3-in-1

    All seed images for a prompt share the same filename prefix in the flat
    samples/ folder; the eval scripts score every image they find there.
    """
    t2i_root = Path(t2i_root).resolve()
    samples_dir = t2i_root / "examples" / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)

    print(f"  Staging images into {samples_dir} …")
    staged = []
    for i, (seed_imgs, prompt) in enumerate(zip(grouped_images, prompts)):
        prefix = _safe_prompt_for_filename(prompt)
        for j, img_path in enumerate(seed_imgs):
            # Naming: {prompt_prefix}_{prompt_idx:06d}_{seed:04d}{ext}
            dst = samples_dir / f"{prefix}_{i:06d}_{j:04d}{img_path.suffix}"
            shutil.copy2(img_path, dst)
            staged.append(dst)

    results = {}

    def _run(label, cmd, cwd):
        ret = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"  [WARNING] {label} exited {ret.returncode}: {ret.stderr[-1500:]}")
        return ret.returncode == 0

    # ---- Non-spatial: BLIP-VQA (color / shape / texture attribute binding) ----
    print("  Running T2I-CompBench BLIP-VQA (non-spatial) …")
    if _run("BLIP-VQA",
            [sys.executable, "BLIP_vqa.py", "--out_dir=../examples/"],
            cwd=t2i_root / "BLIPvqa_eval"):
        rfile = t2i_root / "examples" / "annotation_blip" / "vqa_result.json"
        if rfile.exists():
            results["non_spatial_blip_vqa"] = json.loads(rfile.read_text())

    # ---- Non-spatial: CLIPScore (text–image alignment) ----------------------
    print("  Running T2I-CompBench CLIPScore (non-spatial) …")
    if _run("CLIPScore",
            [sys.executable, "CLIPScore_eval/CLIP_similarity.py", "--outpath=examples/"],
            cwd=t2i_root):
        rfile = t2i_root / "examples" / "annotation_clip" / "vqa_result.json"
        if rfile.exists():
            results["non_spatial_clip"] = json.loads(rfile.read_text())

    # ---- Spatial: UniDet 2D + 3D --------------------------------------------
    print("  Running T2I-CompBench UniDet 2D spatial …")
    _run("UniDet-2D", [sys.executable, "2D_spatial_eval.py"],
         cwd=t2i_root / "UniDet_eval")

    print("  Running T2I-CompBench UniDet 3D spatial …")
    _run("UniDet-3D", [sys.executable, "3D_spatial_eval.py"],
         cwd=t2i_root / "UniDet_eval")

    print("  Running T2I-CompBench UniDet numeracy …")
    _run("UniDet-numeracy", [sys.executable, "numeracy_eval.py"],
         cwd=t2i_root / "UniDet_eval")

    # ---- Complex: 3-in-1 ----------------------------------------------------
    _3in1_data = t2i_root / "examples" / "dataset" / "complex_val.txt"
    _complex_prompts = set(l.strip().lower() for l in _3in1_data.read_text().splitlines()) if _3in1_data.exists() else set()
    _our_prompts = set(p.strip().lower() for p in prompts)
    if not _complex_prompts or not (_our_prompts & _complex_prompts):
        print("  Skipping T2I-CompBench 3-in-1 (prompts are not from the complex_val set) …")
    elif _run("3-in-1",
            [sys.executable, "3_in_1.py", "--outpath=../examples/"],
            cwd=t2i_root / "3_in_1_eval"):
        rfile = t2i_root / "examples" / "annotation_3_in_1" / "vqa_result.json"
        if rfile.exists():
            results["complex_3in1"] = json.loads(rfile.read_text())

    # Archive result files
    ann_dir = t2i_root / "examples"
    for subdir in ann_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("annotation_"):
            shutil.copytree(subdir, output_dir / "t2icompbench" / subdir.name,
                            dirs_exist_ok=True)

    for f in staged:
        f.unlink(missing_ok=True)

    return results


# ---------------------------------------------------------------------------
# GenEval
# ---------------------------------------------------------------------------

def run_geneval(grouped_images, prompts, geneval_root: str, model_path: str,
                geneval_tasks: list[str] | None, output_dir: Path):
    """
    GenEval: object-detection-based text-image alignment using Mask2Former.

    Per metrics.md, evaluates: two-object, counting, color, color_attribution.
    All seed images for each prompt are placed in one subfolder so
    evaluate_images.py scores them together (standard protocol).

    Default (no --geneval_tasks_file): every prompt is evaluated under ALL FOUR
    task types.  This creates 4 × N subfolders in the staging directory — one
    per (prompt, task) combination — and reports per-task accuracy across all
    prompts.

    Requires CUDA (evaluate_images.py asserts DEVICE == "cuda").
    """
    geneval_root = Path(geneval_root).resolve()
    eval_script = geneval_root / "evaluation" / "evaluate_images.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"GenEval evaluate_images.py not found at {eval_script}")

    tmp_dir = output_dir / "_geneval_staging"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build the flat list of (seed_imgs, prompt, task) entries to stage.
    # When no tasks file is given: expand each prompt × all 4 task types so
    # that every image is evaluated under two_obj / counting / color /
    # color_attribution, matching the full GenEval evaluation protocol.
    if geneval_tasks is not None:
        entries = list(zip(grouped_images, prompts, geneval_tasks))
    else:
        entries = [
            (seed_imgs, prompt, task)
            for seed_imgs, prompt in zip(grouped_images, prompts)
            for task in GENEVAL_TASKS
        ]

    print(f"  Staging {len(entries)} evaluations "
          f"({len(prompts)} prompt(s) × "
          f"{len(entries) // len(prompts)} task(s), "
          f"{len(grouped_images[0])} seed(s) each) …")

    for i, (seed_imgs, prompt, task) in enumerate(entries):
        subdir = tmp_dir / f"{i:05d}"
        samples = subdir / "samples"
        samples.mkdir(parents=True, exist_ok=True)

        for j, img_path in enumerate(seed_imgs):
            from PIL import Image as _PIL
            _PIL.open(img_path).convert("RGB").save(samples / f"{j:04d}.png")

        meta = {"prompt": prompt, "tag": task, "include_types": [task]}
        (subdir / "metadata.jsonl").write_text(json.dumps(meta) + "\n")

    results_file = output_dir / "geneval_results.jsonl"

    print("  Running GenEval evaluate_images.py …")
    ret = subprocess.run(
        [
            sys.executable, str(eval_script),
            str(tmp_dir),
            "--outfile", str(results_file),
            "--model-path", str(model_path),
        ],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  [WARNING] GenEval exited {ret.returncode}:\n{ret.stderr[:600]}")
        shutil.rmtree(tmp_dir)
        return {}

    summary_script = geneval_root / "evaluation" / "summary_scores.py"
    summary_ret = subprocess.run(
        [sys.executable, str(summary_script), str(results_file)],
        capture_output=True, text=True,
    )

    summary = {}
    import re as _re
    for line in summary_ret.stdout.splitlines():
        # per-task line: "two_obj          = 75.00% (6 / 8)"
        m = _re.match(r"^(\S+)\s*=\s*([\d.]+)%", line)
        if m:
            summary[m.group(1)] = float(m.group(2)) / 100
        # overall line: "Overall score (avg. over tasks): 0.75000"
        m = _re.search(r"Overall score.*?:\s*([\d.]+)", line)
        if m:
            summary["overall"] = float(m.group(1))

    print(f"  GenEval overall: {summary.get('overall', 'N/A')}")

    shutil.rmtree(tmp_dir)
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified IQA evaluation: ImageReward, CLIP-IQA, NIQE, T2I-CompBench, GenEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--images_dir", required=True,
                   help="Directory of images to evaluate")
    p.add_argument("--prompts_file",
                   help="Text file with one prompt per line — one line per image "
                        "(flat layout) or one line per subfolder (folder layout)")
    p.add_argument("--output_dir", default="./iqa_results",
                   help="Directory to write result JSON files (default: ./iqa_results)")
    p.add_argument("--metrics", nargs="+",
                   choices=["image_reward", "clipiqa", "niqe", "t2icompbench", "geneval"],
                   default=["clipiqa", "niqe"],
                   help="Which metrics to run (default: clipiqa niqe)")
    p.add_argument("--t2i_root", default="./T2I-CompBench",
                   help="Path to cloned T2I-CompBench repo (default: ./T2I-CompBench)")
    p.add_argument("--geneval_root", default="./geneval",
                   help="Path to cloned GenEval repo (default: ./geneval)")
    p.add_argument("--geneval_model_path", default="./geneval/evaluation/models",
                   help="Path to GenEval Mask2Former model weights")
    p.add_argument("--geneval_tasks_file",
                   help="Path to a JSON file: either a list of task strings "
                        '["two_obj","color",...] (one per prompt) or '
                        'a dict {"img.png": "two_obj", ...}. '
                        "Valid tasks: two_obj, counting, color, color_attribution. "
                        "If omitted, tasks are inferred from prompt text.")
    p.add_argument("--device", default="cuda",
                   help="PyTorch device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    prompt_metrics = {"image_reward", "t2icompbench", "geneval"}
    needs_prompts = prompt_metrics & set(args.metrics)
    if needs_prompts and not args.prompts_file:
        sys.exit(f"ERROR: --prompts_file is required for: {sorted(needs_prompts)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, prompts, grouped = load_images_and_prompts(
        args.images_dir, args.prompts_file
    )
    print(f"Images: {len(images)}, prompt groups: {len(grouped)}, "
          f"seeds per group: {len(grouped[0])}")

    all_results: dict = {}

    # ---- ImageReward -------------------------------------------------------
    if "image_reward" in args.metrics:
        print("\n=== ImageReward ===")
        all_results["image_reward"] = run_image_reward(grouped, prompts, device)

    # ---- CLIP-IQA ----------------------------------------------------------
    if "clipiqa" in args.metrics:
        print("\n=== CLIP-IQA ===")
        all_results["clipiqa"] = run_clipiqa(images, device)

    # ---- NIQE --------------------------------------------------------------
    if "niqe" in args.metrics:
        print("\n=== NIQE ===")
        all_results["niqe"] = run_niqe(images, device)

    # ---- T2I-CompBench -----------------------------------------------------
    if "t2icompbench" in args.metrics:
        print("\n=== T2I-CompBench ===")
        t2i_root = Path(args.t2i_root)
        if not t2i_root.exists():
            print(f"  [SKIP] T2I-CompBench repo not found at {t2i_root}. Run setup.sh first.")
        else:
            all_results["t2icompbench"] = run_t2icompbench(
                grouped, prompts, t2i_root, output_dir
            )

    # ---- GenEval -----------------------------------------------------------
    if "geneval" in args.metrics:
        print("\n=== GenEval ===")
        geneval_root = Path(args.geneval_root)
        if not geneval_root.exists():
            print(f"  [SKIP] GenEval repo not found at {geneval_root}. Run setup.sh first.")
        else:
            geneval_tasks = None
            if args.geneval_tasks_file:
                raw = json.loads(Path(args.geneval_tasks_file).read_text())
                if isinstance(raw, list):
                    geneval_tasks = raw  # ["two_obj", "color", ...]
                else:
                    # dict keyed by first-seed image filename
                    geneval_tasks = [
                        raw.get(group[0].name, "two_obj") for group in grouped
                    ]
            all_results["geneval"] = run_geneval(
                grouped, prompts, geneval_root,
                args.geneval_model_path, geneval_tasks, output_dir
            )

    # ---- Save & summarize --------------------------------------------------
    results_file = output_dir / "iqa_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {results_file}")

    print("\n=== Summary ===")
    for metric, result in all_results.items():
        if not isinstance(result, dict):
            continue
        if "mean" in result:
            tag = "(lower=better)" if result.get("lower_better") else ""
            print(f"  {metric:20s} mean = {result['mean']:.4f}  {tag}")
        elif metric == "geneval" and "summary" in result:
            s = result["summary"]
            overall = s.get('overall', 'N/A')
            overall_str = f"{overall:.4f}" if isinstance(overall, (int, float)) else str(overall)
            print(f"  {'geneval':20s} overall = {overall_str}")
            for task in GENEVAL_TASKS:
                if task in s:
                    print(f"    {task:22s} = {s[task]:.4f}")
        elif metric == "t2icompbench":
            for sub in result:
                print(f"  t2icompbench/{sub}")


if __name__ == "__main__":
    main()

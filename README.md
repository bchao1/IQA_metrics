# IQA Metrics

Unified evaluation script for image quality assessment (IQA) metrics used in text-to-image generation research.

## Metrics

| Metric | Type | Reference | Better |
|---|---|---|---|
| `image_reward` | Text-conditioned human-preference score | [ImageReward](https://github.com/zai-org/ImageReward) | Higher |
| `clipiqa` | No-reference perceptual quality (CLIP-IQA) | [PyIQA](https://github.com/chaofengc/IQA-PyTorch) | Higher |
| `niqe` | No-reference natural image quality | [PyIQA](https://github.com/chaofengc/IQA-PyTorch) | Lower |
| `t2icompbench` | Compositional text-image alignment (spatial / non-spatial / complex) | [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) | Higher |
| `geneval` | Object-detection-based alignment (two-object, counting, color, color attribution) | [GenEval](https://github.com/djghosh13/geneval) | Higher |

## Requirements

- Python 3.10+
- PyTorch with CUDA (GenEval **requires** a CUDA GPU)
- ~10 GB disk space for model weights

## Setup

Run the setup script once from the `IQA_metrics` directory. It installs all dependencies, clones required repos, and downloads model weights.

```bash
bash setup.sh
```

This will:
1. Install `pyiqa` (CLIP-IQA, NIQE) and `image-reward`
2. Clone and configure T2I-CompBench with UniDet, GLIP, and MiDaS weights
3. Clone GenEval and download Mask2Former weights

## Image Directory Layouts

The script auto-detects one of two layouts:

**Flat layout** — one image per prompt, all in one directory:
```
images_dir/
  img000.jpg
  img001.jpg
  img002.jpg
```

**Folder layout** — multiple seeds per prompt, one subdirectory per prompt:
```
images_dir/
  00000/
    0000.jpg
    0001.jpg
    0002.jpg
    0003.jpg
  00001/
    0000.jpg
    ...
```

The folder layout is the standard protocol for T2I-CompBench and GenEval (4 seeds per prompt).

## Usage

### No-reference quality only (no prompts needed)

```bash
python evaluate_iqa.py \
    --images_dir /path/to/images \
    --metrics clipiqa niqe \
    --output_dir ./results
```

### Full evaluation on a flat dataset (e.g. COCO 5k)

```bash
python evaluate_iqa.py \
    --images_dir /local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val_images \
    --prompts_file /local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val.txt \
    --metrics image_reward clipiqa niqe t2icompbench geneval \
    --output_dir ./results
```

### Folder layout with 4 seeds per prompt

```bash
python evaluate_iqa.py \
    --images_dir /path/to/images_by_prompt/ \
    --prompts_file /path/to/prompts.txt \
    --metrics image_reward clipiqa niqe t2icompbench geneval \
    --output_dir ./results
```

### GenEval with explicit per-prompt task types

```bash
# tasks.json contains a list like ["two_obj", "color", "counting", ...]
python evaluate_iqa.py \
    --images_dir /path/to/images_by_prompt/ \
    --prompts_file /path/to/prompts.txt \
    --metrics geneval \
    --geneval_tasks_file tasks.json \
    --output_dir ./results
```

Valid GenEval task types: `two_obj`, `counting`, `color`, `color_attribution`.

### Run individual metrics

```bash
# ImageReward only
python evaluate_iqa.py \
    --images_dir /path/to/images \
    --prompts_file /path/to/prompts.txt \
    --metrics image_reward

# T2I-CompBench only
python evaluate_iqa.py \
    --images_dir /path/to/images \
    --prompts_file /path/to/prompts.txt \
    --metrics t2icompbench
```

## Quick Test

The included test script runs all metrics on a 10-image subset:

```bash
# Edit IMAGES_DIR and PROMPTS_FILE in run_test_eval.sh to point to your data
bash run_test_eval.sh
```

Output is saved to `./test_eval/`.

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--images_dir` | *(required)* | Directory of images to evaluate |
| `--prompts_file` | — | Text file with one prompt per line (required for `image_reward`, `t2icompbench`, `geneval`) |
| `--metrics` | `clipiqa niqe` | One or more of: `image_reward clipiqa niqe t2icompbench geneval` |
| `--output_dir` | `./iqa_results` | Directory for output JSON files |
| `--device` | `cuda` | PyTorch device: `cuda`, `cpu`, `cuda:0`, `auto` |
| `--t2i_root` | `./T2I-CompBench` | Path to cloned T2I-CompBench repo |
| `--geneval_root` | `./geneval` | Path to cloned GenEval repo |
| `--geneval_model_path` | `./geneval/evaluation/models` | Path to GenEval Mask2Former weights |
| `--geneval_tasks_file` | — | JSON file mapping prompts to GenEval task types |

## Output

Results are written to `{output_dir}/iqa_results.json` with per-image and per-prompt scores:

```json
{
  "image_reward": {
    "mean": 0.312,
    "per_prompt": { "a cat on a mat": 0.421, ... },
    "per_image": { "img000.jpg": 0.421, ... }
  },
  "clipiqa": {
    "mean": 0.654,
    "per_image": { "img000.jpg": 0.701, ... },
    "lower_better": false
  },
  "niqe": {
    "mean": 4.23,
    "per_image": { "img000.jpg": 3.91, ... },
    "lower_better": true
  },
  "geneval": {
    "summary": {
      "overall": 0.75,
      "two_obj": 0.80,
      "counting": 0.65,
      "color": 0.78,
      "color_attribution": 0.77
    }
  }
}
```

A summary is also printed to stdout at the end of each run.

## Example Dataset

- Images: `/local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val_images` (5000 images)
- Prompts: `/local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val.txt` (5000 prompts)

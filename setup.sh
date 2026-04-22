#!/usr/bin/env bash
# Setup script for all IQA metric libraries.
# Run once from the IQA_metrics directory before using evaluate_iqa.py.
#
# NOTE: GenEval (evaluate_images.py) requires a CUDA GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
echo "=== Installing PyIQA (CLIP-IQA, NIQE) ==="
pip install pyiqa

echo "=== Installing ImageReward ==="
pip install image-reward

# ImageReward's BLIP/med.py uses apply_chunking_to_forward which was moved
# out of transformers.modeling_utils in transformers>=4.36. Patch the import.
python3 - <<'PYEOF'
import re, site, os
for sp in site.getsitepackages():
    med = os.path.join(sp, "ImageReward", "models", "BLIP", "med.py")
    if not os.path.exists(med):
        continue
    src = open(med).read()
    old = ("from transformers.modeling_utils import (\n"
           "    PreTrainedModel,\n"
           "    apply_chunking_to_forward,\n"
           "    find_pruneable_heads_and_indices,\n"
           "    prune_linear_layer,\n"
           ")")
    new = ("from transformers.modeling_utils import PreTrainedModel\n"
           "try:\n"
           "    from transformers.modeling_utils import (\n"
           "        apply_chunking_to_forward,\n"
           "        find_pruneable_heads_and_indices,\n"
           "        prune_linear_layer,\n"
           "    )\n"
           "except ImportError:\n"
           "    from transformers.pytorch_utils import (\n"
           "        apply_chunking_to_forward,\n"
           "        find_pruneable_heads_and_indices,\n"
           "        prune_linear_layer,\n"
           "    )")
    if old in src:
        open(med, "w").write(src.replace(old, new))
        print(f"Patched {med}")
    else:
        print(f"Patch not needed or already applied: {med}")
    break
PYEOF

# ---------------------------------------------------------------------------
echo "=== Cloning T2I-CompBench ==="
if [ ! -d "T2I-CompBench" ]; then
    git clone https://github.com/Karine-Huang/T2I-CompBench.git
fi

echo "=== Installing T2I-CompBench dependencies ==="
# OpenAI CLIP (not on PyPI, required by CLIPScore and 3-in-1 evals)
pip install git+https://github.com/openai/CLIP.git

# fvcore + iopath must be installed before detectron2
pip install fvcore iopath

# detectron2 requires torch at build time.
# --no-build-isolation makes the already-installed torch visible to the build env.
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13"

# Install remaining T2I-CompBench packages, filtering out entries that would
# downgrade torch, torchvision, transformers, and other packages already
# present at newer compatible versions.
python3 - <<'PYEOF'
import re, sys

skip_prefixes = [
    "torch==", "torchvision==", "triton==", "nvidia-",
    "accelerate==", "datasets==", "transformers==",
    "huggingface-hub==", "tokenizers==", "safetensors==",
    "pillow==", "numpy==", "tqdm==", "requests==", "filelock==",
    "packaging==", "pyyaml==", "regex==", "fsspec==",
    "multiprocess==", "dill==", "pyarrow==", "xxhash==",
    "sympy==", "networkx==", "typing_extensions==", "typing-extensions==",
    "jinja2==", "markupsafe==",
    "aiohttp==", "aiosignal==", "frozenlist==", "multidict==", "yarl==",
    "certifi==", "charset-normalizer==", "idna==", "urllib3==",
    "psutil==", "six==", "pytz==", "python-dateutil==",
    "pandas==", "scipy==", "matplotlib==", "opencv-python==",
    # Would downgrade packages needed by pyiqa / image-reward or the existing env
    "timm==", "tensorboard==", "tensorboard-data-server==",
    "grpcio==", "protobuf==", "google-auth==", "google-auth-oauthlib==",
    "fairscale==",
    # Would downgrade core packaging / test infrastructure
    "requests==", "tqdm==", "filelock==", "platformdirs==",
    "pluggy==", "importlib-metadata==", "importlib-resources==", "zipp==",
    # spaCy ecosystem: built against old numpy; install latest below instead
    "spacy==", "spacy-legacy==", "spacy-loggers==", "thinc==",
    "blis==", "preshed==", "cymem==", "murmurhash==",
    "catalogue==", "srsly==", "wasabi==", "confection==",
    "langcodes==", "pathy==", "typer==",
    # git-sourced packages handled above
]

kept = []
for raw in open("T2I-CompBench/requirements.txt"):
    line = raw.strip()
    if not line or line.startswith("#") or line.startswith("git+") or line.startswith("en-core-web"):
        continue
    if any(line.lower().startswith(p.lower()) for p in skip_prefixes):
        continue
    kept.append(line)

out = "/tmp/t2i_reqs_filtered.txt"
with open(out, "w") as f:
    f.write("\n".join(kept) + "\n")
print(f"Filtered to {len(kept)} packages (wrote {out})")
PYEOF
pip install -r /tmp/t2i_reqs_filtered.txt

# Install spaCy >=3.7 (required for numpy 2.x / thinc 8.3+) and its English model
pip install "spacy>=3.7"
python3 -m spacy download en_core_web_sm

# ---------------------------------------------------------------------------
echo "=== Downloading T2I-CompBench UniDet model weights ==="
UNIDET_WEIGHTS="T2I-CompBench/UniDet_eval/experts/expert_weights"
mkdir -p "$UNIDET_WEIGHTS"

# Unified multi-domain object detector
if [ ! -f "$UNIDET_WEIGHTS/Unified_learned_OCIM_RS200_6x+2x.pth" ]; then
    wget -q --show-progress -P "$UNIDET_WEIGHTS" \
        "https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth"
fi
# MiDaS depth estimator (3D spatial eval)
if [ ! -f "$UNIDET_WEIGHTS/dpt_hybrid-midas-501f0c75.pt" ]; then
    wget -q --show-progress -P "$UNIDET_WEIGHTS" \
        "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt"
fi
# GLIP large model
if [ ! -f "$UNIDET_WEIGHTS/glip_large_model.pth" ]; then
    pip install gdown
    gdown "https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq" \
        -O "$UNIDET_WEIGHTS/glip_large_model.pth"
fi

# ---------------------------------------------------------------------------
echo "=== Cloning GenEval ==="
if [ ! -d "geneval" ]; then
    git clone https://github.com/djghosh13/geneval.git
fi

echo "=== Installing GenEval Python dependencies ==="
# open_clip and clip-benchmark are imported directly by evaluate_images.py
pip install open-clip-torch clip-benchmark

# mmcv-full 1.x is required by mmdetection 2.x.
# --no-build-isolation is needed so the build can find the installed torch.
pip install openmim
pip install --no-build-isolation "mmcv-full>=1.7.0,<2.0"

echo "=== Installing mmdetection 2.x (Mask2Former backend for GenEval) ==="
if [ ! -d "mmdetection" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
fi
cd mmdetection
git checkout 2.x
# setup.py develop works with older setup.py-based packages that
# pip install -e cannot handle with newer pip (missing build_editable hook)
python setup.py develop
cd "$SCRIPT_DIR"

echo "=== Downloading GenEval Mask2Former weights ==="
mkdir -p geneval/evaluation/models
cd geneval
bash evaluation/download_models.sh "evaluation/models/"
cd "$SCRIPT_DIR"

echo ""
echo "Setup complete."
echo "Usage:"
echo "  python evaluate_iqa.py \\"
echo "      --images_dir /path/to/images \\"
echo "      --prompts_file /path/to/prompts.txt \\"
echo "      --metrics image_reward clipiqa niqe t2icompbench geneval"

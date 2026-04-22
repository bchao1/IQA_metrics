#!/usr/bin/env bash
# Run all IQA metrics on the first 10 coco5k images.
# Output saved to ./test_eval/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGES_DIR=/local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val_images
PROMPTS_FILE=/local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val.txt

# --- build 10-image subset ---
mkdir -p test_images
for i in $(seq -w 0 9); do
    ln -sf "$IMAGES_DIR/0000${i}.jpg" "test_images/0000${i}.jpg"
done

head -10 "$PROMPTS_FILE" > test_prompts.txt

# --- run all metrics ---
python3 evaluate_iqa.py \
    --images_dir test_images \
    --prompts_file test_prompts.txt \
    --metrics t2icompbench \
    --output_dir test_eval

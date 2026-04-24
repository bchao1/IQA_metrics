#!/usr/bin/env bash
# Run all IQA metrics on images. Results saved to <images_dir>/eval/
# Usage: ./run_test_eval.sh [images_dir ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PROMPTS_FILE=/local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val.txt

# Default image folders (used when no args are passed)
DEFAULT_DIRS=(
    #/local/howard/efficient_diffusion/wavelet_diffusion/results/0.01_L2_7steps/dct
    #/local/howard/efficient_diffusion/wavelet_diffusion/results/0.05_L2_7steps/dct
    #/local/howard/efficient_diffusion/wavelet_diffusion/results/0.05_L2_7steps/dwt
    #/local/howard/efficient_diffusion/wavelet_diffusion/results/0.1_L2_7steps/dct
    #/local/howard/efficient_diffusion/wavelet_diffusion/results/fullres_7steps
    #/local/howard/efficient_diffusion/wavelet_diffusion/results/0.05_L3/dct
    #/local/howard/efficient_diffusion/wavelet_diffusion/results/0.01_L2/dct
    /local/howard/efficient_diffusion/wavelet_diffusion/results/NFE_Controlled/0.01_L2_7steps/dct
    /local/howard/efficient_diffusion/wavelet_diffusion/results/NFE_Controlled/0.01_L2_10steps/dct
)

if [ "$#" -gt 0 ]; then
    DIRS=("$@")
else
    DIRS=("${DEFAULT_DIRS[@]}")
fi

for IMAGES_DIR in "${DIRS[@]}"; do
    echo ""
    echo "=== Evaluating: $IMAGES_DIR ==="
    python3 evaluate_iqa.py \
        --images_dir "$IMAGES_DIR" \
        --prompts_file "$PROMPTS_FILE" \
        --metrics image_reward clipiqa niqe \
        --output_dir "$IMAGES_DIR/eval"
done

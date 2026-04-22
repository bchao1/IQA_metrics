# IQA scripts

- ImageReward: https://github.com/zai-org/ImageReward
- ClipIQA: In IQA-Pytorch
    - We utilize CLIP-IQA provided by PyIQA with the default prompt setting.
- NIQE: In IQA-Pytorch
    - We utilize NIQE provided by PyIQA with the default prompt setting.
- T2ICompbench: https://github.com/Karine-Huang/T2I-CompBench
    - For evaluation, we measured performance on spatial, non-spatial and complex sets. For quantitative evaluation, each prompt is sampled with four different random seeds.
- GenEval: https://github.com/djghosh13/geneval
    - We use two-object, counting, color, and color attribution prompts to evaluate models. In our experiments, each prompt is sampled with four different random seeds.
- IQA-Pytorch: https://github.com/chaofengc/IQA-PyTorch

# Example dataset
- `/local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val_images` with 5000 images
- `/local/howard/efficient_diffusion/wavelet_diffusion/data/coco5k_val.txt ` with 5000 prompts
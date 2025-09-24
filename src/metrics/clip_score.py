from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image

from src.preprocess.clip import encode_images, encode_texts, get_clip_model_and_preprocess


def clip_scores_for_prompts(image_paths: list[Path], prompt: str, device: str | None = None) -> torch.Tensor:
    model, preprocess, tokenizer, dev = get_clip_model_and_preprocess(device=device)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_features = encode_images(images, model, preprocess, dev)
    text_features = encode_texts([prompt], model, tokenizer, dev)
    sims = (image_features @ text_features.t()).squeeze(1)
    return sims.detach().cpu()


def aggregate_clip_scores_by_class(
    gen_root: Path, class_to_prompt: Dict[str, str], device: str | None = None
) -> Dict[str, Tuple[float, float, float]]:
    results: Dict[str, Tuple[float, float, float]] = {}
    for class_name, prompt in class_to_prompt.items():
        img_paths = sorted([p for p in (gen_root / class_name).glob("*.png")])
        if not img_paths:
            continue
        sims = clip_scores_for_prompts(img_paths, prompt, device=device)
        mean = float(sims.mean().item())
        low = float(torch.quantile(sims, 0.025).item())
        high = float(torch.quantile(sims, 0.975).item())
        results[class_name] = (mean, low, high)
    return results

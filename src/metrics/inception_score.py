from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models

from src.preprocess.dataset import load_pil_image
from src.preprocess.fid_is import preprocess_batch


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_inception_for_logits(device: str) -> torch.nn.Module:
    # torchvision enforces aux_logits=True when using pretrained weights
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.eval()
    if device in {"cuda", "mps"}:
        model = model.to(device)
    return model


def _predict_probs(images: list[Image.Image], model: torch.nn.Module, device: str, batch_size: int = 32) -> np.ndarray:
    probs_list: list[np.ndarray] = []
    i = 0
    softmax = torch.nn.Softmax(dim=1)
    while i < len(images):
        batch = images[i : i + batch_size]
        i += batch_size
        x = preprocess_batch(batch, image_size=299)
        if device in {"cuda", "mps"}:
            x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            p = softmax(logits)
        probs_list.append(p.detach().cpu().numpy())
    return np.concatenate(probs_list, axis=0)


def inception_score(images: list[Image.Image], splits: int = 10, batch_size: int = 32) -> Tuple[float, float]:
    device = _pick_device()
    model = _load_inception_for_logits(device)
    preds = _predict_probs(images, model, device, batch_size)
    N = preds.shape[0]
    if N == 0:
        return float("nan"), float("nan")
    # Ensure non-empty splits
    # If there are fewer images than desired splits, use a single split to avoid
    # the degenerate case where each split contains exactly 1 image, which forces IS=1.
    if N < splits:
        splits = 1
    indices = np.array_split(np.arange(N), splits)
    scores = []
    for idx in indices:
        part = preds[idx]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl)))
    return float(np.mean(scores)), float(np.std(scores))


def inception_score_for_paths(image_paths: list[Path], splits: int = 10, batch_size: int = 32) -> Tuple[float, float]:
    images = [load_pil_image(p) for p in image_paths]
    return inception_score(images, splits=splits, batch_size=batch_size)

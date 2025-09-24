from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models

from src.preprocess.dataset import index_dataset, load_pil_image, stratified_real_vs_real_split
from src.preprocess.fid_is import preprocess_batch


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_inception(device: str) -> torch.nn.Module:
    # torchvision enforces aux_logits=True when using pretrained weights
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.eval()
    if device in {"cuda", "mps"}:
        model = model.to(device)
    return model


def _compute_activations(
    images: list[Image.Image], model: torch.nn.Module, device: str, batch_size: int = 32
) -> np.ndarray:
    feats: list[np.ndarray] = []
    i = 0
    while i < len(images):
        batch = images[i : i + batch_size]
        i += batch_size
        x = preprocess_batch(batch, image_size=299)
        if device in {"cuda", "mps"}:
            x = x.to(device)
        with torch.no_grad():
            y = model(x)
        y_np = y.detach().cpu().numpy()
        feats.append(y_np)
    return np.concatenate(feats, axis=0)


def _stats(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = activations.mean(axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    # Symmetrize then take eigen decomposition; clamp eigenvalues to >= 0 for numerical stability
    a_sym = (a + a.T) * 0.5
    w, v = np.linalg.eigh(a_sym)
    w_clamped = np.clip(w, 0.0, None)
    sqrt_w = np.sqrt(w_clamped)
    return (v * sqrt_w) @ v.T


def _fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    s1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    s2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    covmean = _sqrtm_psd(s1.dot(s2))
    fid = float(diff.dot(diff) + np.trace(s1 + s2 - 2.0 * covmean))
    return fid


def compute_fid_for_paths(real_paths: list[Path], gen_paths: list[Path], batch_size: int = 32) -> float:
    device = _pick_device()
    model = _load_inception(device)
    real_images = [load_pil_image(p) for p in real_paths]
    gen_images = [load_pil_image(p) for p in gen_paths]
    real_act = _compute_activations(real_images, model, device, batch_size)
    gen_act = _compute_activations(gen_images, model, device, batch_size)
    mu_r, sig_r = _stats(real_act)
    mu_g, sig_g = _stats(gen_act)
    return _fid(mu_r, sig_r, mu_g, sig_g)


def compute_fid_with_ci(
    real_paths: list[Path], gen_paths: list[Path], num_bootstrap: int = 200, batch_size: int = 32, seed: int = 0
) -> tuple[float, float, float]:
    device = _pick_device()
    model = _load_inception(device)
    real_images = [load_pil_image(p) for p in real_paths]
    gen_images = [load_pil_image(p) for p in gen_paths]
    real_act = _compute_activations(real_images, model, device, batch_size)
    gen_act = _compute_activations(gen_images, model, device, batch_size)
    mu_r, sig_r = _stats(real_act)
    mu_g, sig_g = _stats(gen_act)
    base = _fid(mu_r, sig_r, mu_g, sig_g)

    if num_bootstrap is None or num_bootstrap <= 0:
        return float(base), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = gen_act.shape[0]
    samples = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        mu_g_b = gen_act[idx].mean(axis=0)
        sig_g_b = np.cov(gen_act[idx], rowvar=False)
        samples[i] = _fid(mu_r, sig_r, mu_g_b, sig_g_b)
    low = float(np.quantile(samples, 0.025))
    high = float(np.quantile(samples, 0.975))
    return float(base), low, high


def collect_paths_by_class(
    real_root: Path, gen_root: Path, class_name: str | None = None
) -> tuple[list[Path], list[Path]]:
    if class_name is None:
        real_paths = sorted(
            [p for p in real_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        gen_paths = sorted([p for p in gen_root.rglob("*.png")])
        return real_paths, gen_paths
    real_dir = real_root / class_name
    gen_dir = gen_root / class_name
    real_paths = sorted([p for p in real_dir.iterdir() if p.is_file()])
    gen_paths = sorted([p for p in gen_dir.iterdir() if p.is_file()])
    return real_paths, gen_paths


def real_vs_real_baseline(real_root: Path, test_fraction: float = 0.5, seed: int = 0) -> float:
    records = index_dataset(real_root)
    train_recs, test_recs = stratified_real_vs_real_split(records, test_fraction=test_fraction, seed=seed)
    left = [r.path for r in train_recs]
    right = [r.path for r in test_recs]
    return compute_fid_for_paths(left, right)

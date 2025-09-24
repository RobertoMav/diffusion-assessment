from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from tqdm import tqdm

from src.metrics.fid import collect_paths_by_class, compute_fid_with_ci, real_vs_real_baseline
from src.metrics.inception_score import inception_score_for_paths
from src.preprocess.clip import encode_images, encode_texts, get_clip_model_and_preprocess
from src.utils.io import ensure_dir, get_output_paths, read_classes_file, read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics for a run")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--classes-file", type=Path, default=Path("src/configs/classes.txt"))
    parser.add_argument("--prompts-file", type=Path, default=Path("src/configs/prompts.yaml"))
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap samples for FID confidence intervals (0 disables CI)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = read_classes_file(args.classes_file)
    prompts_cfg = read_yaml(args.prompts_file)
    class_to_prompt: Dict[str, str] = {c: str(prompts_cfg["prompts"].get(c, f"a photo of a {c}")) for c in classes}

    paths = get_output_paths(args.run_id)
    gen_root = paths["generated_root"]
    real_root = Path("data/real")

    ensure_dir(paths["metrics_root"])
    per_class_csv = paths["metrics_root"] / "per_class.csv"
    overall_csv = paths["metrics_root"] / "overall.csv"
    clip_per_class_csv = paths["metrics_root"] / "clip_per_class.csv"
    clip_overall_csv = paths["metrics_root"] / "clip_overall.csv"

    # Stage 1: Per-class FID only
    with per_class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "fid", "fid_low", "fid_high"])
        for c in tqdm(classes, desc="FID per-class"):
            real_paths, gen_paths = collect_paths_by_class(real_root, gen_root, c)
            print(f"passed {c}")
            if not real_paths or not gen_paths:
                continue
            fid, fid_low, fid_high = compute_fid_with_ci(real_paths, gen_paths, num_bootstrap=args.bootstrap)
            writer.writerow([c, fid, fid_low, fid_high])

    # Stage 1: Overall FID and IS, real-vs-real baseline
    overall_rows = []
    real_paths_all, gen_paths_all = collect_paths_by_class(real_root, gen_root, None)
    if real_paths_all and gen_paths_all:
        fid, fid_low, fid_high = compute_fid_with_ci(real_paths_all, gen_paths_all, num_bootstrap=args.bootstrap)
        is_mean, is_std = inception_score_for_paths(gen_paths_all)
        rr_fid = real_vs_real_baseline(real_root)
        overall_rows.append(["overall", fid, fid_low, fid_high, is_mean, is_std, rr_fid])

    with overall_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "fid", "fid_low", "fid_high", "is_mean", "is_std", "real_vs_real_fid"])
        for row in overall_rows:
            writer.writerow(row)

    # Stage 2: CLIPScore (separate files), with progress bars
    model, preprocess, tokenizer, dev = get_clip_model_and_preprocess()
    all_sims: list[torch.Tensor] = []

    with clip_per_class_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "clip_mean", "clip_low", "clip_high"])
        for c in tqdm(classes, desc="CLIPScore per-class"):
            img_paths = sorted((gen_root / c).glob("*.png"))
            if not img_paths:
                continue
            images = [Image.open(p).convert("RGB") for p in img_paths]
            image_features = encode_images(images, model, preprocess, dev)
            text_features = encode_texts([class_to_prompt[c]], model, tokenizer, dev)
            sims = (image_features @ text_features.t()).squeeze(1).detach().cpu()
            all_sims.append(sims)
            mean = float(sims.mean().item())
            low = float(torch.quantile(sims, 0.025).item())
            high = float(torch.quantile(sims, 0.975).item())
            writer.writerow([c, mean, low, high])

    if all_sims:
        sims_cat = torch.cat(all_sims, dim=0)
        with clip_overall_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scope", "clip_mean", "clip_low", "clip_high"])
            writer.writerow(
                [
                    "overall",
                    float(sims_cat.mean().item()),
                    float(torch.quantile(sims_cat, 0.025).item()),
                    float(torch.quantile(sims_cat, 0.975).item()),
                ]
            )


if __name__ == "__main__":
    main()

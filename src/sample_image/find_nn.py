from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from src.preprocess.clip import encode_images, get_clip_model_and_preprocess
from src.utils.io import ensure_dir, get_output_paths, read_classes_file


def embed_images(paths: list[Path], model, preprocess, device: str, batch_size: int = 64) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in paths]
    feats: List[torch.Tensor] = []
    i = 0
    while i < len(images):
        chunk = images[i : i + batch_size]
        i += batch_size
        feats.append(encode_images(chunk, model, preprocess, device))
    return torch.cat(feats, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find CLIP nearest real neighbor for one generated image per class")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--classes-file", type=Path, default=Path("src/configs/classes.txt"))
    args = parser.parse_args()

    classes = read_classes_file(args.classes_file)
    paths = get_output_paths(args.run_id)
    ensure_dir(paths["nearest_root"])

    model, preprocess, tokenizer, device = get_clip_model_and_preprocess()

    # cache real embeddings
    real_root = Path("data/real")
    all_real_paths: list[Path] = sorted(
        [p for p in real_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    cache_path = paths["embeddings_root"] / "clip_real.pt"
    ensure_dir(cache_path.parent)
    if cache_path.exists():
        cached = torch.load(cache_path)
        real_paths = [Path(p) for p in cached["paths"]]
        real_embeds = cached["embeds"]
    else:
        real_paths = all_real_paths
        real_embeds = embed_images(real_paths, model, preprocess, device)
        torch.save({"paths": [str(p) for p in real_paths], "embeds": real_embeds}, cache_path)

    for c in classes:
        gen_dir = paths["generated_root"] / c
        gen_imgs = sorted(gen_dir.glob("*.png"))
        if not gen_imgs:
            continue
        gen_path = gen_imgs[0]
        gen_embed = embed_images([gen_path], model, preprocess, device)
        sims = (gen_embed @ real_embeds.t()).squeeze(0)
        idx = int(torch.argmax(sims).item())
        nn_path = real_paths[idx]

        # build side-by-side
        left = Image.open(gen_path).convert("RGB")
        right = Image.open(nn_path).convert("RGB")
        h = max(left.height, right.height)
        new_w = left.width + right.width
        canvas = Image.new("RGB", (new_w, h), color=(255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        out_path = paths["nearest_root"] / f"{c}.png"
        canvas.save(str(out_path))


if __name__ == "__main__":
    main()

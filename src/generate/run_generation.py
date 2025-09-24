from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from src.utils.io import ensure_dir, get_output_paths, read_classes_file, read_yaml, write_json
from src.utils.seed import set_seed

from .model import load_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images for classes and seeds")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier; default uses timestamp")
    parser.add_argument("--classes-file", type=Path, default=Path("src/configs/classes.txt"))
    parser.add_argument("--prompts-file", type=Path, default=Path("src/configs/prompts.yaml"))
    parser.add_argument("--run-config", type=Path, default=Path("src/configs/run.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_cfg = read_yaml(args.run_config)
    prompts_cfg = read_yaml(args.prompts_file)
    classes = read_classes_file(args.classes_file)

    run_id = args.run_id or str(int(time.time()))
    paths = get_output_paths(run_id)
    ensure_dir(paths["generated_root"])
    ensure_dir(paths["logs_root"])

    pipe, device = load_pipeline(
        model_id=run_cfg.get("model_id", "runwayml/stable-diffusion-v1-5"),
        device=run_cfg.get("device", "auto"),
        precision=run_cfg.get("precision", "auto"),
        scheduler=run_cfg.get("scheduler", "dpmpp_2m"),
    )

    seeds: List[int] = list(map(int, run_cfg.get("seeds", [0])))
    steps: int = int(run_cfg.get("num_inference_steps", 35))
    guidance: float = float(run_cfg.get("guidance_scale", 7.0))
    batch_size: int = int(run_cfg.get("batch_size", 4))
    images_per_class: int = int(run_cfg.get("images_per_class", 64))
    image_size: int = int(run_cfg.get("image_size", 512))

    negative_prompt: str = str(prompts_cfg.get("negative_prompt", ""))
    class_to_prompt: Dict[str, str] = {c: str(prompts_cfg["prompts"].get(c, f"a photo of a {c}")) for c in classes}

    # Save run log
    versions: Dict[str, Any] = {
        "diffusers": __import__("diffusers").__version__,
        "torch": __import__("torch").__version__,
        "transformers": __import__("transformers").__version__,
    }
    write_json(
        paths["logs_root"] / "config.json",
        {"run": run_cfg, "prompts": prompts_cfg, "versions": versions, "device": device},
    )

    for class_name in classes:
        out_dir = paths["generated_root"] / class_name
        ensure_dir(out_dir)

        prompt = class_to_prompt[class_name]
        remaining = images_per_class
        for seed in seeds:
            if remaining <= 0:
                break
            set_seed(seed)
            to_make = min(batch_size, remaining)
            prompts = [prompt] * to_make
            negs = [negative_prompt] * to_make if negative_prompt else None
            result = pipe(
                prompt=prompts,
                negative_prompt=negs,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=image_size,
                width=image_size,
            )
            images: List[Image.Image] = list(result.images)
            for idx, im in enumerate(images):
                filename = f"{class_name}_{seed}_{idx:04d}.png"
                im.save(str(out_dir / filename))
            remaining -= len(images)


if __name__ == "__main__":
    main()

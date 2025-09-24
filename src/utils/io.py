from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    return data


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def read_classes_file(classes_file: Path) -> list[str]:
    class_names: list[str] = []
    with classes_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            class_names.append(line)
    if not class_names:
        raise ValueError(f"No classes found in {classes_file}")
    return class_names


def save_image(pil_image: Image.Image, destination_path: Path) -> None:
    ensure_dir(destination_path.parent)
    pil_image.save(str(destination_path))


def get_output_paths(run_id: str) -> dict[str, Path]:
    root = Path("outputs")
    paths: dict[str, Path] = {
        "generated_root": root / "generated" / run_id,
        "metrics_root": root / "metrics" / run_id,
        "embeddings_root": root / "embeddings",
        "nearest_root": root / "nearest_neighbors" / run_id,
        "logs_root": root / "logs" / run_id,
    }
    return paths


def list_image_files(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

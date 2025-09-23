from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    class_name: str


def iter_image_files(root: Path, allowed_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_exts:
            yield path


def index_dataset(real_root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for class_dir in sorted([p for p in real_root.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        for img_path in iter_image_files(class_dir):
            records.append(ImageRecord(path=img_path, class_name=class_name))
    return records


def stratified_real_vs_real_split(
    records: Sequence[ImageRecord],
    test_fraction: float = 0.5,
    seed: int = 0,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    rng = random.Random(seed)

    # Group by class
    class_to_records: Dict[str, List[ImageRecord]] = {}
    for rec in records:
        class_to_records.setdefault(rec.class_name, []).append(rec)

    train_records: List[ImageRecord] = []
    test_records: List[ImageRecord] = []
    for class_name, class_records in class_to_records.items():
        shuffled = class_records[:]
        rng.shuffle(shuffled)
        split_index = int(round(len(shuffled) * (1.0 - test_fraction)))
        train_records.extend(shuffled[:split_index])
        test_records.extend(shuffled[split_index:])

    return train_records, test_records


def load_pil_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

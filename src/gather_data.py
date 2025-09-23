import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image
from torchvision.datasets import Food101
from tqdm import tqdm


def read_classes_file(classes_file: Path) -> List[str]:
    class_names: List[str] = []
    with classes_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            class_names.append(line)
    if not class_names:
        raise ValueError(f"No classes found in {classes_file}")
    return class_names


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def try_copy_from_source_path(source_path: Optional[Path], destination_path: Path) -> bool:
    if source_path is None:
        return False
    try:
        shutil.copyfile(str(source_path), str(destination_path))
        return True
    except Exception:
        return False


def save_pil(image: Image.Image, destination_path: Path, resize_to: Optional[int] = None) -> None:
    rgb_image = image.convert("RGB")
    if resize_to is not None:
        rgb_image = rgb_image.resize((resize_to, resize_to), Image.BICUBIC)
    rgb_image.save(str(destination_path), format="JPEG", quality=95)


def collect_food101_images(
    classes: List[str],
    output_root: Path,
    download_root: Path,
    min_per_class: int,
    max_per_class: Optional[int],
    resize_to: Optional[int],
) -> Dict[str, int]:
    selected_classes: Set[str] = set(classes)
    counts: Dict[str, int] = {c: 0 for c in selected_classes}

    # Ensure destination directories exist
    for class_name in selected_classes:
        ensure_dir(output_root / class_name)

    # Load both train and test splits to maximize available images per class
    datasets: List[Tuple[str, Food101]] = []
    for split in ("train", "test"):
        ds = Food101(root=str(download_root), split=split, download=True)
        datasets.append((split, ds))

    for split_name, dataset in datasets:
        # class_to_idx maps class_name -> label_index
        class_to_index: Dict[str, int] = dataset.class_to_idx  # type: ignore[attr-defined]
        index_to_class: List[str] = [None] * len(class_to_index)
        for name, idx in class_to_index.items():
            index_to_class[idx] = name

        image_files: Optional[List[str]] = getattr(dataset, "_image_files", None)

        with tqdm(total=len(dataset), desc=f"Processing {split_name}") as pbar:
            for i in range(len(dataset)):
                # (image, target)
                pil_image, label_index = dataset[i]
                class_name = index_to_class[label_index]

                if class_name not in selected_classes:
                    pbar.update(1)
                    continue

                if max_per_class is not None and counts[class_name] >= max_per_class:
                    pbar.update(1)
                    continue

                destination_dir = output_root / class_name
                ensure_dir(destination_dir)

                # Construct destination filename; try to preserve original basename if we know source path
                original_path: Optional[Path] = None
                if image_files is not None:
                    try:
                        original_path = Path(image_files[i])
                    except Exception:
                        original_path = None

                if original_path is not None:
                    base_name = original_path.stem
                else:
                    base_name = f"{split_name}_{i:06d}"

                destination_path = destination_dir / f"{base_name}.jpg"
                # Avoid collisions
                suffix_id = 1
                while destination_path.exists():
                    destination_path = destination_dir / f"{base_name}_{suffix_id}.jpg"
                    suffix_id += 1

                copied_ok = False
                if resize_to is None and original_path is not None and original_path.exists():
                    # Best: byte-for-byte copy if no resizing requested
                    copied_ok = try_copy_from_source_path(original_path, destination_path)

                if not copied_ok:
                    save_pil(pil_image, destination_path, resize_to=resize_to)

                counts[class_name] += 1
                pbar.update(1)

    # Check minimums
    for class_name in selected_classes:
        if counts[class_name] < min_per_class:
            print(f"Warning: class '{class_name}' gathered {counts[class_name]} images (< {min_per_class}).")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather Food-101 images into data/real/<class>/")
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=Path("src/configs/classes.txt"),
        help="Path to classes.txt (one class per line)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/real"),
        help="Directory to write per-class folders",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path("data"),
        help="Root path for torchvision to download Food-101",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=250,
        help="Warn if fewer images than this are gathered per class",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on number of images per class",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optionally resize saved images to RESIZE x RESIZE (e.g., 512)",
    )
    args = parser.parse_args()

    classes = read_classes_file(args.classes_file)
    ensure_dir(args.out_dir)
    ensure_dir(args.download_root)

    counts = collect_food101_images(
        classes=classes,
        output_root=args.out_dir,
        download_root=args.download_root,
        min_per_class=args.min_per_class,
        max_per_class=args.max_per_class,
        resize_to=args.resize,
    )

    print("\nGathered image counts per class:")
    for class_name in classes:
        print(f"- {class_name}: {counts.get(class_name, 0)}")


if __name__ == "__main__":
    main()

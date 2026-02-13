"""Download and prepare the Kaggle tomato ripeness dataset.

Downloads the 'enalis/tomatoes-dataset' from Kaggle using kagglehub,
then organises the images into the flat folder structure expected by
our training pipeline:

    data/tomato/
    ├── unripe/
    ├── ripe/
    ├── old/
    └── damaged/

Usage:
    python -m ml_training.vision.download_dataset
    python -m ml_training.vision.download_dataset --output data/tomato
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


KAGGLE_DATASET = "enalis/tomatoes-dataset"

# Map possible folder-name variants (lowercase) to our canonical names.
# We'll also strip leading/trailing whitespace from discovered folder names.
CANONICAL_CLASSES = ["unripe", "ripe", "old", "damaged"]
CLASS_ALIASES: dict[str, str] = {
    "unripe": "unripe",
    "un_ripe": "unripe",
    "un-ripe": "unripe",
    "green": "unripe",
    "ripe": "ripe",
    "red": "ripe",
    "old": "old",
    "overripe": "old",
    "over_ripe": "old",
    "damaged": "damaged",
    "damage": "damaged",
    "defective": "damaged",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dirs(root: Path) -> dict[str, Path]:
    """Walk *root* and return {canonical_class: path} for each class found."""
    found: dict[str, Path] = {}

    for candidate in sorted(root.rglob("*")):
        if not candidate.is_dir():
            continue
        # Check if this directory contains actual images
        has_images = any(
            f.suffix.lower() in VALID_EXTENSIONS
            for f in candidate.iterdir()
            if f.is_file()
        )
        if not has_images:
            continue

        name = candidate.name.strip().lower().replace(" ", "_")
        canonical = CLASS_ALIASES.get(name)
        if canonical and canonical not in found:
            found[canonical] = candidate
            print(f"  Found class '{canonical}' at: {candidate}")

    return found


def copy_images(src_dir: Path, dst_dir: Path) -> int:
    """Copy image files from *src_dir* into *dst_dir*. Returns count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_dir.iterdir()):
        if img.is_file() and img.suffix.lower() in VALID_EXTENSIONS:
            dst = dst_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            count += 1
    return count


def download_and_prepare(output_dir: str | Path = "data/tomato") -> Path:
    """Download the Kaggle dataset and organise into class folders.

    Args:
        output_dir: Where to place the final class-organised images.

    Returns:
        Path to the output directory.
    """
    output_path = Path(output_dir)

    # Check if already prepared
    existing = [
        c for c in CANONICAL_CLASSES if (output_path / c).is_dir()
    ]
    if len(existing) == len(CANONICAL_CLASSES):
        total = sum(
            len(list((output_path / c).glob("*")))
            for c in CANONICAL_CLASSES
        )
        if total > 100:
            print(f"✓ Dataset already exists at {output_path} ({total} images)")
            for c in CANONICAL_CLASSES:
                n = len(list((output_path / c).glob("*")))
                print(f"  {c}: {n}")
            return output_path

    # Download via kagglehub
    print(f"Downloading '{KAGGLE_DATASET}' from Kaggle...")
    try:
        import kagglehub
        download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        print(f"Downloaded to: {download_path}")
    except Exception as e:
        print(f"Error downloading with kagglehub: {e}")
        print("\nFallback: please download manually from:")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
        print(f"Extract to: {output_path}")
        raise SystemExit(1)

    # Discover class subdirectories in the download
    print("\nScanning for class directories...")
    class_dirs = find_class_dirs(download_path)

    if not class_dirs:
        print(f"\nERROR: Could not find any recognised class directories in {download_path}")
        print("Directory structure:")
        for p in sorted(download_path.rglob("*"))[:30]:
            rel = p.relative_to(download_path)
            prefix = "📁" if p.is_dir() else "📄"
            print(f"  {prefix} {rel}")
        raise SystemExit(1)

    missing = [c for c in CANONICAL_CLASSES if c not in class_dirs]
    if missing:
        print(f"\n⚠ WARNING: Missing classes: {missing}")
        print("Available classes:", list(class_dirs.keys()))

    # Copy to output directory
    print(f"\nCopying images to {output_path}...")
    total = 0
    for cls_name, src in class_dirs.items():
        # The dataset may have separate train/val dirs. Merge them.
        # First copy from the found directory
        n = copy_images(src, output_path / cls_name)
        total += n
        print(f"  {cls_name}: {n} images")

    # Also check for any train/val split structure and merge
    for split_name in ["train", "training", "val", "validation", "test"]:
        split_root = download_path
        for candidate in download_path.rglob(split_name):
            if candidate.is_dir():
                split_root = candidate
                extra_dirs = find_class_dirs(split_root)
                for cls_name, src in extra_dirs.items():
                    if cls_name in CANONICAL_CLASSES:
                        n = copy_images(src, output_path / cls_name)
                        if n > 0:
                            total += n
                            print(f"  {cls_name} (+{n} from {split_name}/)")

    print(f"\n✅ Dataset prepared: {total} total images in {output_path}")
    for c in CANONICAL_CLASSES:
        if (output_path / c).is_dir():
            n = len(list((output_path / c).glob("*")))
            print(f"  {c}: {n}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle tomato dataset")
    parser.add_argument(
        "--output", type=str, default="data/tomato",
        help="Output directory for organised images (default: data/tomato)",
    )
    args = parser.parse_args()
    download_and_prepare(args.output)


if __name__ == "__main__":
    main()

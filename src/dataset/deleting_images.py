import shutil
from pathlib import Path

import cv2
import numpy as np


def filter_augmented_images(
    source_dir: str | Path,
    discard_dir: str | Path,
    min_threshold: int = 230,
    max_area_ratio: float = 0.20,
) -> None:
    source_path = Path(source_dir)
    discard_path = Path(discard_dir)

    discard_path.mkdir(parents=True, exist_ok=True)

    # Track statistics per disease class
    class_stats: dict[str, dict[str, int]] = {}
    total_scanned = 0
    total_discarded = 0

    print(f"Scanning dataset in: {source_path}\n")

    for disease_folder in source_path.iterdir():
        if not disease_folder.is_dir():
            continue

        class_name = disease_folder.name
        class_stats[class_name] = {"scanned": 0, "discarded": 0}

        # Create matching class folder in the discard directory
        class_discard_dir = discard_path / class_name
        class_discard_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing class: {class_name}...")

        for img_path in disease_folder.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue

            class_stats[class_name]["scanned"] += 1
            total_scanned += 1

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            img_h, img_w = image.shape
            image_area = img_h * img_w

            # Isolate the white artifacts
            _, bright_mask = cv2.threshold(image, min_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            is_augmented = False

            for contour in contours:
                if cv2.contourArea(contour) > image_area * max_area_ratio:
                    is_augmented = True
                    break

            # Move the image to the corresponding class discard folder
            if is_augmented:
                target_path = class_discard_dir / img_path.name
                shutil.move(str(img_path), str(target_path))

                class_stats[class_name]["discarded"] += 1
                total_discarded += 1

    print("\n" + "=" * 50)
    print("Curation Summary")
    print("=" * 50)

    for cls_name, stats in class_stats.items():
        scanned = stats["scanned"]
        discarded = stats["discarded"]
        remaining = scanned - discarded
        print(
            f"- {cls_name:<15} {remaining:>5} remaining  (Discarded {discarded}/{scanned})"
        )

    print("-" * 50)
    print(f"TOTAL SCANNED:   {total_scanned}")
    print(f"TOTAL DISCARDED: {total_discarded}")
    print(f"TOTAL REMAINING: {total_scanned - total_discarded}")


if __name__ == "__main__":
    DATASET_SOURCE = (
        "/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli/train"
    )
    DISCARD_FOLDER = "/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli-discarded/train"

    filter_augmented_images(DATASET_SOURCE, DISCARD_FOLDER)

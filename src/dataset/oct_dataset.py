import concurrent.futures
from pathlib import Path
from typing import Literal, Generator
import numpy as np
from PIL import Image

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


class OctDataset:
    allowed_image_extensions: set[str] = {".jpg", ".jpeg", ".png"}

    def __init__(self, dataset_path: Path, allowed_classes: list[str]):
        self.path = dataset_path
        self.allowed_classes = allowed_classes

    def get_class_mapping(self, allowed_classes: list[str]) -> dict[str, int]:
        return {class_name: idx for idx, class_name in enumerate(allowed_classes)}

    def _process_image(
        self, image_path: Path, label_idx: int, class_name: str
    ) -> tuple[np.ndarray, int, str]:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((512, 512))
            arr = np.array(img)
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
        return arr, label_idx, class_name

    def _get_image_paths(self, split_path: Path) -> list[tuple[Path, int, str]]:
        class_mapping = self.get_class_mapping(self.allowed_classes)
        image_paths: list[tuple[Path, int, str]] = []
        for class_name, label_idx in class_mapping.items():
            class_dir = split_path / class_name
            if not class_dir.exists():
                continue

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in self.allowed_image_extensions:
                    image_paths.append((image_path, label_idx, class_name))

        if not image_paths:
            print(f"Warning: No images found for split '{split_path}'.")

        return image_paths

    def get_image_batch(
        self, dataset_split_name: Literal["train", "val", "test"], batch_size: int
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        split_path = self.path / dataset_split_name
        image_paths = self._get_image_paths(split_path)

        batch_images, batch_labels = [], []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[yellow]{task.fields[current_class]}"),
            TimeRemainingColumn(),
        ) as progress:
            extraction_task = progress.add_task(
                f"[green]Extracting {dataset_split_name.upper()}",
                total=len(image_paths),
                current_class="",
            )

            current_displayed_class = ""

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_img = {
                    executor.submit(self._process_image, path, label, class_name): path
                    for path, label, class_name in image_paths
                }

                # As tasks complete (regardless of order), yield them
                for future in concurrent.futures.as_completed(future_to_img):
                    try:
                        image_array, label_idx, class_name = future.result()

                        if class_name != current_displayed_class:
                            progress.update(
                                extraction_task, current_class=f"({class_name})"
                            )
                            current_displayed_class = class_name

                        batch_images.append(image_array)
                        batch_labels.append(label_idx)

                        if len(batch_images) == batch_size:
                            yield (
                                np.array(batch_images, dtype=np.uint8),
                                np.array(batch_labels, dtype=np.uint8),
                            )
                            batch_images, batch_labels = [], []

                    except Exception as e:
                        failed_path = future_to_img[future]
                        progress.console.print(
                            f"[red]Failed to process {failed_path.name}: {e}"
                        )

                    progress.advance(extraction_task)

        # Yield remainder
        if batch_images:
            yield (
                np.array(batch_images, dtype=np.uint8),
                np.array(batch_labels, dtype=np.uint8),
            )

    def gather_full_split(
        self,
        dataset_split_name: Literal["train", "val", "test"],
        samples_per_class: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size = 32
        generator = self.get_image_batch(dataset_split_name, batch_size)

        X_batches, y_batches = [], []
        for X_batch, y_batch in generator:
            X_batches.append(X_batch)
            y_batches.append(y_batch)

        if not X_batches:
            raise ValueError(
                f"No data could be gathered for split '{dataset_split_name}'."
            )

        return (
            np.concatenate(X_batches, axis=0),
            np.concatenate(y_batches, axis=0),
        )

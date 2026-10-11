import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import (
    OctDataset,
)
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.registry import PreprocessingRegistry


def pregenerate_dataset(
    raw_dataset_path: Path,
    target_dataset_path: Path,
    preprocessor: PreprocessingPipeline,
    allowed_classes: list[str],
    max_images_per_class: int | None = None,
):
    """
    Applies the preprocessing pipeline to all images and saves them losslessly to a new directory.
    Leverages OctDataset for patient-stratified round-robin sampling.
    """
    print(f"Source: {raw_dataset_path}")
    print(f"Target: {target_dataset_path}")

    target_dataset_path.mkdir(parents=True, exist_ok=True)

    print("Indexing dataset for patient-stratified extraction...")
    dataset = OctDataset(
        dataset_path=raw_dataset_path,
        allowed_classes=allowed_classes,
        samples_per_class=max_images_per_class,
        preprocessor=None,
    )

    tasks: list[tuple[Path, Path]] = []

    for src_path, class_name in zip(dataset.image_paths, dataset.class_names):
        target_class_dir = target_dataset_path / class_name
        target_class_dir.mkdir(exist_ok=True)

        target_path = target_class_dir / f"{src_path.stem}.png"
        tasks.append((src_path, target_path))

    if not tasks:
        print("No images found to process.")
        return

    def process_and_save(src: Path, dst: Path) -> None:
        if dst.exists():
            return

        with Image.open(src) as img:
            img = img.convert("L")
            arr = np.array(img)

            if preprocessor is not None:
                arr = preprocessor.execute(arr)

            arr = np.ascontiguousarray(arr, dtype=np.uint8)

            # Save losslessly.
            out_img = Image.fromarray(arr)
            out_img.save(dst, format="PNG")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("[green]Pregenerating dataset...", total=len(tasks))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_and_save, src, dst) for src, dst in tasks
            ]

            for _ in concurrent.futures.as_completed(futures):
                progress.advance(task_id)

    print("\nPregeneration complete. Dataset is ready for tuning.")


if __name__ == "__main__":
    config = OctConfig.from_yaml("config/run.yaml")

    preprocessing_pipeline = PreprocessingRegistry.create_pipeline(
        config.preprocessing_pipeline
    )

    raw_path = Path(config.paths.original_dataset)

    target_path = raw_path.parent / f"{raw_path.name}_preprocessed"

    pregenerate_dataset(
        raw_dataset_path=raw_path,
        target_dataset_path=target_path,
        preprocessor=preprocessing_pipeline,
        allowed_classes=config.dataset.classes,
        max_images_per_class=None,
    )

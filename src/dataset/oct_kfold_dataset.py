import concurrent.futures
import re
import uuid
from collections.abc import Generator
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
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing.preprocessing_pipeline import PreprocessingPipeline


class OctDatasetCrossVal:
    allowed_image_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(
        self,
        dataset_path: Path,
        allowed_classes: list[str],
        samples_per_class: int | None = None,
        preprocessor: PreprocessingPipeline | None = None,
        n_splits: int = 5,
        n_folds_to_train: int | None = None,
        validation_fold: int | None = None,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.path = dataset_path
        self.allowed_classes = allowed_classes
        self.preprocessor = preprocessor
        self.n_splits = n_splits

        self.n_folds_to_train = n_folds_to_train
        self.validation_fold = validation_fold

        if validation_fold is not None and not 1 <= validation_fold <= n_splits:
            raise ValueError(f"validation_fold must be between 1 and {n_splits}.")

        if n_folds_to_train is not None and not 1 <= n_folds_to_train <= n_splits:
            raise ValueError(f"n_folds_to_train must be between 1 and {n_splits}.")

        self.shuffle = shuffle

        self.random_state = random_state

        (
            self.all_image_paths,
            self.all_labels,
            self.all_class_names,
            self.all_patient_ids,
        ) = self._collect_all_images(samples_per_class)

    @property
    def class_mapping(self) -> dict[str, int]:
        """Maps each class name to its numeric label."""
        return {class_name: idx for idx, class_name in enumerate(self.allowed_classes)}

    def _extract_patient_id(self, path: Path) -> str:
        match = re.match(r"^[^-]+-(\d+)-\d+$", path.stem)

        if match is None:
            random_id = f"random_group_{uuid.uuid4().hex[:8]}"
            print(
                f"Warning: Unexpected format for {path.name}. Assigned to {random_id}"
            )
            return random_id

        return match.group(1)

    def _collect_all_images(
        self,
        samples_per_class: int | None = None,
    ) -> tuple[
        list[Path],
        np.ndarray,
        list[str],
        np.ndarray,
    ]:
        image_paths: list[Path] = []
        labels: list[int] = []
        class_names: list[str] = []
        patient_ids: list[str] = []

        for class_name, label_idx in self.class_mapping.items():
            class_dir = self.path / class_name

            if not class_dir.exists():
                print(f"Warning: Disease directory '{class_dir}' does not exist.")
                continue

            valid_samples = 0

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() not in self.allowed_image_extensions:
                    continue

                if samples_per_class is not None and valid_samples >= samples_per_class:
                    break

                image_paths.append(image_path)
                labels.append(label_idx)
                class_names.append(class_name)
                patient_ids.append(self._extract_patient_id(image_path))

                valid_samples += 1

        return (
            image_paths,
            np.array(labels, dtype=np.uint8),
            class_names,
            np.array(patient_ids),
        )

    def _process_image(
        self,
        image_path: Path,
        label_idx: int,
        class_name: str,
    ) -> tuple[np.ndarray, int, str]:
        with Image.open(image_path) as img:
            img = img.convert("L")
            arr = np.array(img)

            if self.preprocessor is not None:
                arr = self.preprocessor.execute(arr)

            arr = np.ascontiguousarray(arr, dtype=np.uint8)

        return arr, label_idx, class_name

    def get_folds(
        self,
    ) -> Generator[
        tuple[int, np.ndarray, np.ndarray],
        None,
        None,
    ]:
        """
        Generates fold_idx, train_indices, and val_indices for the active folds.

        Yields
        ------
        tuple[int, np.ndarray, np.ndarray]
            (fold_idx, train_indices, val_indices)
        """
        cv = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=(self.random_state if self.shuffle else None),
        )

        all_splits = list(
            cv.split(
                self.all_image_paths,
                self.all_labels,
                groups=self.all_patient_ids,
            )
        )

        # Determine which fold indices to generate based on config
        if self.validation_fold is not None:
            folds_to_run = [self.validation_fold - 1]
        else:
            folds_to_run = list(range(self.n_splits))
            if self.n_folds_to_train is not None:
                folds_to_run = folds_to_run[: self.n_folds_to_train]

        for fold_idx in folds_to_run:
            val_indices = all_splits[fold_idx][1]
            train_indices = np.concatenate(
                [all_splits[i][1] for i in range(self.n_splits) if i != fold_idx]
            )

            # Patient leakage assertion check
            train_patients = set(self.all_patient_ids[train_indices])
            val_patients = set(self.all_patient_ids[val_indices])
            overlap = train_patients & val_patients

            if overlap:
                raise RuntimeError(f"Patient leakage detected: {sorted(overlap)[:10]}")

            yield fold_idx, train_indices, val_indices

    def get_fold_image_batch(
        self,
        indices: np.ndarray,
        batch_size: int,
        description: str = "Processing Fold",
    ) -> Generator[
        tuple[np.ndarray, np.ndarray],
        None,
        None,
    ]:
        fold_paths = [self.all_image_paths[i] for i in indices]
        fold_labels = [self.all_labels[i] for i in indices]
        fold_classes = [self.all_class_names[i] for i in indices]

        batch_images: list[np.ndarray] = []
        batch_labels: list[int] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[yellow]{task.fields[current_class]}"),
            TimeRemainingColumn(),
        ) as progress:
            extraction_task = progress.add_task(
                f"[green]{description}",
                total=len(fold_paths),
                current_class="",
            )

            current_displayed_class = ""

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_img = {
                    executor.submit(
                        self._process_image,
                        path,
                        label,
                        class_name,
                    ): path
                    for path, label, class_name in zip(
                        fold_paths,
                        fold_labels,
                        fold_classes,
                    )
                }

                for future in concurrent.futures.as_completed(future_to_img):
                    failed_path = future_to_img[future]

                    try:
                        image_array, label_idx, class_name = future.result()

                        if class_name != current_displayed_class:
                            progress.update(
                                extraction_task,
                                current_class=f"({class_name})",
                            )
                            current_displayed_class = class_name

                        batch_images.append(image_array)
                        batch_labels.append(label_idx)

                        if len(batch_images) == batch_size:
                            yield (
                                np.array(batch_images, dtype=np.uint8),
                                np.array(batch_labels, dtype=np.uint8),
                            )
                            batch_images = []
                            batch_labels = []

                    except Exception as e:
                        progress.console.print(
                            f"[red]Failed to process {failed_path.name}: {e}"
                        )
                    finally:
                        del future_to_img[future]

                    progress.advance(extraction_task)

        if batch_images:
            yield (
                np.array(batch_images, dtype=np.uint8),
                np.array(batch_labels, dtype=np.uint8),
            )

    def gather_fold_split(
        self,
        indices: np.ndarray,
        description: str = "Gathering Fold Data",
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size = 32
        generator = self.get_fold_image_batch(
            indices,
            batch_size=batch_size,
            description=description,
        )

        X_batches: list[np.ndarray] = []
        y_batches: list[np.ndarray] = []

        for X_batch, y_batch in generator:
            X_batches.append(X_batch)
            y_batches.append(y_batch)

        if not X_batches:
            raise ValueError("No data could be gathered for the given fold indices.")

        return (
            np.concatenate(X_batches, axis=0).astype(np.float32),
            np.concatenate(y_batches, axis=0),
        )

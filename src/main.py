import argparse
import json
import os
import sys
import time
import tracemalloc
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import psutil
from PIL import Image
from sklearn.metrics import classification_report

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from dataset.oct_kfold_dataset import OctDatasetCrossVal
from feature_extraction.registry import FeatureExtractorRegistry
from pipeline.pipeline_builder import _get_feature_cache_dir, build_pipeline
from preprocessing.registry import PreprocessingRegistry
from visualization.feature_extractor_visualizer import FeatureExtractorVisualizer
from visualization.train_results_visualizer import TrainResultsVisualizer

# TODO: Differentiate drusen and normal
# TODO: Make "something else" class


def launch_training(config: OctConfig):
    pipeline = build_pipeline(config, True, True, True, True)

    dataset = OctDataset(
        config.paths.original_dataset, config.dataset.classes, preprocessor=None
    )

    base_samples = config.dataset.samples_per_class
    train_limit = int(
        base_samples * (config.dataset.sampling_proportions.train / 100.0)
    )
    val_limit = int(base_samples * (config.dataset.sampling_proportions.val / 100.0))

    y_train = dataset.get_split_labels("train", train_limit)
    y_val = dataset.get_split_labels("val", val_limit)

    cache_dir_str = _get_feature_cache_dir(config)

    is_cached = (
        cache_dir_str is not None
        and Path(cache_dir_str).exists()
        and any(Path(cache_dir_str).iterdir())
    )

    preprocessing = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)
    dataset.preprocessor = preprocessing

    if is_cached:
        print(
            f"[Cache Hit] Loading features from {cache_dir_str}. Bypassing image disk I/O."
        )
        # Dummy shapes matching the label lengths.
        X_train = np.zeros((len(y_train), 1, 1), dtype=np.uint8)
        X_val = np.zeros((len(y_val), 1, 1), dtype=np.uint8)
    else:
        print(
            "[Cache Miss] Cache empty or modified. Loading and preprocessing raw images..."
        )

        X_train, _ = dataset.gather_full_split("train", train_limit)
        X_val, _ = dataset.gather_full_split("val", val_limit)

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, config.paths.model_output / "model.joblib")

    y_train_pred = pipeline.predict(X_train)
    print("\nTraining accuracies:")
    print(classification_report(y_train, y_train_pred, zero_division=0))

    preds = pipeline.predict(X_val)
    print("Validation accuracies:")
    print(classification_report(y_val, preds, zero_division=0))

    class_mapping = dataset.class_mapping
    classes = list(class_mapping.keys())
    TrainResultsVisualizer.show_confusion_matrix(y_val, preds, classes)

def launch_training_kfold(
    config: OctConfig,
    n_splits: int = 5,
    validation_fold: int = -1,
    random_state: int = 42,
):
    """
    Train using grouped stratified K-Fold.

    validation_fold:
        None -> run full K-fold cross-validation.
        1..n_splits -> use that fold as validation and all
        remaining folds as training.

    Examples:
        validation_fold=5:
            Train: Folds 1-4
            Validation: Fold 5

        validation_fold=None:
            Fold 1: train 2-5, validate 1
            Fold 2: train 1,3-5, validate 2
            ...
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    preprocessing = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)

    cv_dataset = OctDatasetCrossVal(
        dataset_path=config.paths.original_dataset,
        allowed_classes=config.dataset.classes,
        samples_per_class=config.dataset.samples_per_class,
        preprocessor=preprocessing,
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = list(cv_dataset.get_folds())

    print(
        f"\n[K-Fold Dataset]"
        f"\nSamples: {len(cv_dataset.all_labels)}"
        f"\nFolds: {n_splits}"
        f"\nSeed: {random_state}"
    )

    # ---------------------------------------------------------
    # QUICK MODE
    # ---------------------------------------------------------

    if validation_fold != -1:
        val_fold_idx = validation_fold - 1

        val_indices = folds[val_fold_idx][1]

        train_indices = np.concatenate(
            [folds[i][0] for i in range(n_splits) if i != val_fold_idx]
        )

        print(
            f"\n[QUICK MODE]"
            f"\nTrain: Folds "
            f"{', '.join(str(i + 1) for i in range(n_splits) if i != val_fold_idx)}"
            f"\nValidation: Fold {validation_fold}"
        )

        X_train, y_train = cv_dataset.gather_fold_split(
            train_indices,
            description="Training Data",
        )

        X_val, y_val = cv_dataset.gather_fold_split(
            val_indices,
            description="Validation Data",
        )

        pipeline = build_pipeline(
            config,
            True,
            True,
            True,
            True,
        )

        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)

        print("\nValidation Report:")
        print(
            classification_report(
                y_val,
                val_preds,
                zero_division=0,
            )
        )

        return pipeline

    # ---------------------------------------------------------
    # FULL K-FOLD CV
    # ---------------------------------------------------------

    all_val_preds = []
    all_val_targets = []

    for fold_idx, (train_indices, val_indices) in enumerate(
        folds,
        start=1,
    ):
        print(f"\n==================== FOLD {fold_idx}/{n_splits} ====================")

        X_train, y_train = cv_dataset.gather_fold_split(
            train_indices,
            description=f"Fold {fold_idx} Train Data",
        )

        X_val, y_val = cv_dataset.gather_fold_split(
            val_indices,
            description=f"Fold {fold_idx} Val Data",
        )

        pipeline = build_pipeline(
            config,
            True,
            True,
            True,
            True,
        )

        pipeline.fit(X_train, y_train)

        val_preds = pipeline.predict(X_val)

        print(f"\nFold {fold_idx} Validation Report:")

        print(
            classification_report(
                y_val,
                val_preds,
                zero_division=0,
            )
        )

        all_val_preds.extend(val_preds)
        all_val_targets.extend(y_val)

        fold_model_path = config.paths.model_output / f"model_fold_{fold_idx}.joblib"

        config.paths.model_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            pipeline,
            fold_model_path,
        )

    # ---------------------------------------------------------
    # AGGREGATED CV RESULTS
    # ---------------------------------------------------------

    all_val_targets = np.asarray(all_val_targets)
    all_val_preds = np.asarray(all_val_preds)

    print("\n==================== OVERALL K-FOLD RESULTS ====================")

    print(
        classification_report(
            all_val_targets,
            all_val_preds,
            zero_division=0,
        )
    )

    classes = list(cv_dataset.class_mapping.keys())

    TrainResultsVisualizer.show_confusion_matrix(
        all_val_targets,
        all_val_preds,
        classes,
    )

def launch_inference(config: OctConfig, image_path: Path, model_path: str):
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((256, 256))
            image_to_infere = np.array(img)
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")

    prediction_idx = pipeline.predict([image_to_infere])[0]

    predicted_class = config.dataset.classes[int(prediction_idx)]

    print(f"Predicted Class: {predicted_class}")


def launch_dataset_preprocessing(config: OctConfig, preprocessed_save_path: Path):
    # Implementation here
    pass


def launch_feature_extracting_visualization(config: OctConfig, image_path: Path):
    feature_extractor = FeatureExtractorRegistry.create(
        config.feature_extractor.active, **vars(config.feature_extractor.params)
    )
    visualizer = FeatureExtractorVisualizer(feature_extractor)
    with Image.open(image_path) as file_object:
        visualizer.visualize(np.array(file_object))


def launch_config_schema_generation(output_path: Path):
    with open(output_path, "w") as f:
        f.write(json.dumps(OctConfig.model_json_schema(), indent=4))
    print(f"Schema successfully generated at {output_path}")


def main():
    """
    Setups the CLI environment and orchestrates pipeline execution.
    """
    parser = argparse.ArgumentParser(
        description="OCT Image Classification Pipeline CLI"
    )

    # Global argument: Every command (except generate-schema) needs a config file
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/run.yaml"),
        help="Path to the YAML configuration file.",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # --- Sub-commands ---

    train_parser = subparsers.add_parser(
        "train", help="Trains the model using the active config."
    )

    kfold_parser = subparsers.add_parser(
        "train-kfold",
        help="Trains the model using Stratified K-Fold cross validation.",
    )
    kfold_parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    kfold_parser.add_argument(
        "--validation-fold",
        type=int,
        default=-1,
        help="",
    )

    inference_parser = subparsers.add_parser(
        "inference", help="Run inference on a single image."
    )
    inference_parser.add_argument(
        "--image", type=Path, required=True, help="Path to the image to classify."
    )
    inference_parser.add_argument(
        "--model", type=str, required=True, help="Name or path of the saved model file."
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess-dataset", help="Performs image preprocessing on the whole dataset."
    )
    preprocess_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output path for the normalized dataset. Defaults to config paths.",
    )

    viz_parser = subparsers.add_parser(
        "visualize-features",
        help="Visualizes the output of the active feature extractor.",
    )
    viz_parser.add_argument(
        "--image", type=Path, required=True, help="Path to the image to visualize."
    )

    schema_parser = subparsers.add_parser(
        "generate-schema", help="Generates the JSON schema for the YAML config."
    )
    schema_parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/run_schema.json"),
        help="Output path for the generated schema.",
    )

    args = parser.parse_args()

    # 1. Handle schema generation early (does not require reading a valid YAML config)
    if args.command == "generate-schema":
        launch_config_schema_generation(args.output)
        sys.exit(0)

    # 2. Load the configuration globally
    try:
        config = OctConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Fatal error loading configuration from {args.config}:\n{e}")
        sys.exit(1)

    # 3. Route execution
    match args.command:
        case "train":
            launch_training(config)

        case "train-kfold":
            launch_training_kfold(config, args.n_splits)

        case "inference":
            launch_inference(config, args.image, args.model)

        case "preprocess-dataset":
            # Resolve the output path: prefer CLI argument, fallback to config definition
            save_path = args.output if args.output else config.paths.normalized_dataset
            launch_dataset_preprocessing(config, save_path)

        case "visualize-features":
            launch_feature_extracting_visualization(config, args.image)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()

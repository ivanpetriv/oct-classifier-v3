import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib  # type: ignore
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from dataset.oct_kfold_dataset import OctDatasetCrossVal
from feature_extraction.registry import FeatureExtractorRegistry
from pipeline.pipeline_builder import build_pipeline
from preprocessing.registry import PreprocessingRegistry
from visualization.feature_extractor_visualizer import FeatureExtractorVisualizer
from visualization.train_results_visualizer import TrainResultsVisualizer


def _get_feature_cache_dir(config: OctConfig) -> str | None:
    """
    Generates a unique cache path based on the feature extractor configurations.
    If configurations change or switch, a new subfolder is targeted automatically.
    """
    if not config.feature_extractor or not config.preprocessing_pipeline:
        return None

    # Handle list-based feature extractor configs
    extractor_info = {
        "feature_extractors": config.feature_extractor,
        "preprocessing": config.preprocessing_pipeline,
    }

    # Serialize config using Pydantic's dict conversion for hashing
    config_json = json.dumps(
        config.model_dump(include={"feature_extractor", "preprocessing_pipeline"}),
        sort_keys=True,
        default=str,
    )
    config_hash = hashlib.md5(config_json.encode()).hexdigest()

    cache_path = Path(config.paths.cache) / f"pipeline_cache_{config_hash}"
    return str(cache_path)


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
    n_folds_to_train: int | None = None,
    validation_fold: int | None = None,
    random_state: int = 42,
):
    preprocessing = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)

    dataset = OctDatasetCrossVal(
        dataset_path=config.paths.original_dataset,
        allowed_classes=config.dataset.classes,
        samples_per_class=config.dataset.samples_per_class,
        preprocessor=preprocessing,
        n_splits=n_splits,
        n_folds_to_train=n_folds_to_train,
        validation_fold=validation_fold,
        shuffle=True,
        random_state=random_state,
    )

    active_folds = list(dataset.get_folds())

    print(
        f"\n[K-Fold Dataset]"
        f"\nSamples: {len(dataset.all_labels)}"
        f"\nFolds: {n_splits}"
        f"\nTraining folds: {len(active_folds)}"
        f"\nSeed: {random_state}"
    )

    all_val_preds = []
    all_val_targets = []

    config.paths.model_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for fold_idx, train_indices, val_indices in active_folds:
        print(
            f"\n==================== "
            f"FOLD {fold_idx + 1}/{n_splits} "
            f"===================="
        )

        X_train, y_train = dataset.gather_fold_split(
            train_indices,
            description=f"Fold {fold_idx + 1} Train",
        )

        X_val, y_val = dataset.gather_fold_split(
            val_indices,
            description=f"Fold {fold_idx + 1} Validation",
        )

        pipeline = build_pipeline(
            config,
            True,
            True,
            True,
            True,
        )

        pipeline.fit(X_train, y_train)

        # After fitting, inspect PCA
        pca_step = pipeline.named_steps["dimensionality_reduction"]

        if hasattr(pca_step, "explained_variance_ratio_"):
            cumsum_var = np.cumsum(pca_step.explained_variance_ratio_)

            print("\n=== PCA Variance Analysis ===")
            print(
                f"Variance in {pca_step.n_components} components: {cumsum_var[-1]:.1%}"
            )

            # Find how many components for 80%, 90%, 95%
            for threshold in [0.8, 0.9, 0.95]:
                n_comp_needed = np.argmax(cumsum_var >= threshold) + 1
                print(f"Components for {threshold:.0%} variance: {n_comp_needed}")

            # Plot it
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.plot(cumsum_var[:200])  # First 200 components
            plt.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
            plt.xlabel("PCA Components")
            plt.ylabel("Cumulative Variance Explained")
            plt.title("PCA Variance Profile")
            plt.legend()
            plt.savefig("pca_variance.png")
            plt.close()

        val_preds = pipeline.predict(X_val)

        print(
            classification_report(
                y_val,
                val_preds,
                zero_division=0,
            )
        )

        all_val_targets.extend(y_val)
        all_val_preds.extend(val_preds)

        joblib.dump(
            pipeline,
            config.paths.model_output / f"model_fold_{fold_idx + 1}.joblib",
        )

    if len(active_folds) > 1:
        print("\n==================== OVERALL K-FOLD RESULTS ====================")

        all_val_targets = np.asarray(all_val_targets)
        all_val_preds = np.asarray(all_val_preds)

        print(
            classification_report(
                all_val_targets,
                all_val_preds,
                zero_division=0,
            )
        )

        TrainResultsVisualizer.show_confusion_matrix(
            all_val_targets,
            all_val_preds,
            list(dataset.class_mapping.keys()),
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
    pass


def launch_feature_extracting_visualization(config: OctConfig, image_path: Path):
    if not config.feature_extractor:
        raise ValueError("No feature extractors configured in run.yaml")

    # Visualize each configured feature extractor sequentially
    with Image.open(image_path) as file_object:
        img_arr = np.array(file_object.convert("L"))

        for extractor_item in config.feature_extractor:
            if isinstance(extractor_item, dict):
                extractor_name, extractor_params = next(iter(extractor_item.items()))
            else:
                extractor_name = extractor_item.active
                params_obj = extractor_item.params
                extractor_params = (
                    params_obj.model_dump(exclude_none=True)
                    if hasattr(params_obj, "model_dump")
                    else vars(params_obj)
                )

            print(f"Visualizing extractor: {extractor_name}")
            feature_extractor = FeatureExtractorRegistry.create(
                extractor_name, **extractor_params
            )
            visualizer = FeatureExtractorVisualizer(feature_extractor)
            visualizer.visualize(img_arr)


def launch_config_schema_generation(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = OctConfig.model_json_schema()

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=4, default=str)

    print(f"Schema successfully generated at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OCT Image Classification Pipeline CLI"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/run.yaml"),
        help="Path to the YAML configuration file.",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

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
        "--n-folds-to-train",
        type=int,
        default=5,
        help="Number of cross-validation folds to train on (default: 5).",
    )
    kfold_parser.add_argument(
        "--validation-fold",
        type=int,
        default=None,
        help="Specific fold (1-indexed) to use for validation. If provided, trains 1 fold.",
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

    if args.command == "generate-schema":
        launch_config_schema_generation(args.output)
        sys.exit(0)

    try:
        config = OctConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Fatal error loading configuration from {args.config}:\n{e}")
        sys.exit(1)

    print(args)

    match args.command:
        case "train":
            launch_training(config)

        case "train-kfold":
            launch_training_kfold(
                config,
                args.n_splits,
                args.n_folds_to_train,
                args.validation_fold,
                random_state=config.random_seed,
            )

        case "inference":
            launch_inference(config, args.image, args.model)

        case "preprocess-dataset":
            save_path = args.output if args.output else config.paths.normalized_dataset
            launch_dataset_preprocessing(config, save_path)

        case "visualize-features":
            launch_feature_extracting_visualization(config, args.image)

        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()

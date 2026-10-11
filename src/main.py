import argparse
from contextlib import contextmanager
import json
import sys
from pathlib import Path
import time

import dotenv
import matplotlib
import numpy as np
from PIL import Image

from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from feature_extraction.registry import FeatureExtractorRegistry
from pipeline.cross_validation_runner import CrossValidationRunner
from pipeline.inference import Inferrer
from pipeline.test import Evaluator
from preprocessing.registry import PreprocessingRegistry
from visualization.feature_extractor_visualizer import FeatureExtractorVisualizer

# matplotlib.use("Agg")

dotenv.load_dotenv()


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"\n[{label}] elapsed: {elapsed:.2f}s ({elapsed / 60:.2f} min)")


def launch_training_final(config: OctConfig):
    """
    Trains the final model on 100% of the dataset and saves it to the output path.
    """
    preprocessing = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)

    dataset = OctDataset(
        dataset_path=config.paths.original_dataset,
        allowed_classes=config.dataset.classes,
        samples_per_class=config.dataset.samples_per_class,
        # preprocessor=preprocessing,
        preprocessor=None,
    )

    runner = CrossValidationRunner(
        config=config,
        dataset=dataset,
        n_splits=2,  # Required by init, but ignored by train_final_model
        random_state=config.random_seed,
    )

    trainer = runner.train_final_model(verbose=True)

    # Save the model
    output_dir = config.paths.model_output
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        output_dir
        / f"model_{'_'.join(list(config.feature_extractor[0].keys()))}.joblib"
    )
    trainer.save(model_path)

    print(f"\nFinal model saved to: {model_path}")


def launch_training_kfold(
    config: OctConfig,
    n_splits: int = 5,
    n_folds_to_train: int | None = None,
    random_state: int = 42,
):
    preprocessing = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)

    dataset = OctDataset(
        dataset_path=config.paths.original_dataset,
        allowed_classes=config.dataset.classes,
        samples_per_class=config.dataset.samples_per_class,
        # preprocessor=preprocessing,
        preprocessor=None,
    )

    runner = CrossValidationRunner(
        config=config, dataset=dataset, n_splits=n_splits, random_state=random_state
    )

    runner.run(n_folds_to_train=n_folds_to_train)


def launch_inference(config: OctConfig, image_path: Path, model_path: str):
    inferrer = Inferrer(config, model_path)
    predicted_class = inferrer.predict(image_path)

    print(f"Predicted Class: {predicted_class}")


def launch_test(config: OctConfig, test_dataset_path: Path, model_path: Path):
    """
    Evaluates the finalized production model against a strictly hold-out test dataset.
    """
    evaluator = Evaluator(
        config=config, model_path=model_path, test_dataset_path=test_dataset_path
    )
    evaluator.evaluate()


def launch_dataset_preprocessing(config: OctConfig, preprocessed_save_path: Path):
    pass


def launch_feature_extracting_visualization(config: OctConfig, image_path: Path):
    if not config.feature_extractor:
        raise ValueError("No feature extractors configured in run.yaml")

    extractors = []

    # Gather all configured feature extractors
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

        print(f"Loading extractor: {extractor_name}")
        extractor_instance = FeatureExtractorRegistry.create(
            extractor_name, **extractor_params
        )
        extractors.append(extractor_instance)

    # Render them all simultaneously
    with Image.open(image_path) as file_object:
        img_arr = np.array(file_object.convert("L"))
        visualizer = FeatureExtractorVisualizer(extractors)
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

    train_final_parser = subparsers.add_parser(
        "train-final",
        help="Trains the final production model on 100%% of the dataset.",
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

    test_parser = subparsers.add_parser(
        "test", help="Evaluates a trained model on a hold-out test dataset."
    )
    test_parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Path to the test dataset directory.",
    )
    test_parser.add_argument(
        "--model",
        type=Path,
        default=Path("model_output/model.joblib"),
        help="Path to the saved model.",
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

    with timed(args.command):
        match args.command:
            case "train-final":
                launch_training_final(config)

            case "train-kfold":
                launch_training_kfold(
                    config,
                    args.n_splits,
                    args.n_folds_to_train,
                    random_state=config.random_seed,
                )

            case "inference":
                launch_inference(config, args.image, args.model)

            case "test":
                launch_test(config, args.test_dir, args.model)

            case "preprocess-dataset":
                save_path = (
                    args.output if args.output else config.paths.normalized_dataset
                )
                launch_dataset_preprocessing(config, save_path)

            case "visualize-features":
                launch_feature_extracting_visualization(config, args.image)

            case _:
                parser.print_help()
                sys.exit(1)


if __name__ == "__main__":
    main()


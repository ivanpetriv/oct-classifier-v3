import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from classification.registry import ClassifierRegistry
from config_settings.oct_config import OctConfig
from dataset.oct_dataset import OctDataset
from feature_extraction.registry import FeatureExtractorRegistry
from pipeline.train import Experiment, SklearnTrainer
from preprocessing.registry import PreprocessingRegistry
from visualization.feature_extractor_visualizer import FeatureExtractorVisualizer


def launch_training(config: OctConfig):
    classes = config.dataset.classes
    dataset = OctDataset(config.paths.normalized_dataset, classes)

    feature_extractor = FeatureExtractorRegistry.create(
        config.feature_extractor.active, **vars(config.feature_extractor.params)
    )
    classifier = ClassifierRegistry.create(
        config.classifier.active, **vars(config.classifier.params)
    )

    steps = [("extractor", feature_extractor)]

    match config.dimensionality_reduction.active:
        case "pca":
            pca = PCA(
                n_components=config.dimensionality_reduction.params.n_components,
                whiten=config.dimensionality_reduction.params.whiten,
                random_state=config.random_seed,
            )
            steps.append(("reducer", pca))
        case "none" | _:
            pass

    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps)

    trainer = SklearnTrainer(pipeline)
    experiment = Experiment(dataset, classes)

    result = experiment.run(trainer, tuner=None)
    print(f"Macro F1: {result.macro_f1}")


def launch_inference(config: OctConfig, image_path: Path, model_path: str):
    try:
        trained_model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    try:
        with Image.open(image_path) as img:
            image_to_infere = np.array(img)
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")

    preprocessing_pipeline = PreprocessingRegistry.create_pipeline(
        config.preprocessing_pipeline
    )
    feature_extractor = FeatureExtractorRegistry.create(
        config.feature_extractor.active, **vars(config.feature_extractor.params)
    )

    preprocessed_image = preprocessing_pipeline.execute(image_to_infere)
    image_features = feature_extractor.extract(preprocessed_image)

    features_2d = image_features.reshape(1, -1)

    prediction_idx = trained_model.predict(features_2d)[0]

    classes = config.dataset.classes
    predicted_class = classes[int(prediction_idx)]

    print("\n--- Inference Result ---")
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

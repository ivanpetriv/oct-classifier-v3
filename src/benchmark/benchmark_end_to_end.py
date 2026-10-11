from pathlib import Path

import cv2

cv2.setNumThreads(1)
import pytest

from config_settings.oct_config import OctConfig
from pipeline.inference import Inferrer
from preprocessing.registry import PreprocessingRegistry

IMAGE_DIR = Path(
    "/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli/test/"
)

MODEL_PATH_LDCH = Path("models/model_ldch.joblib")
MODEL_PATH_CDCH = Path("models/model_cdch.joblib")
MODEL_PATH_LBP = Path("models/model_lbp.joblib")
MODEL_PATH_HOG = Path("models/model_hog.joblib")

config_ldch = OctConfig.from_yaml(Path("config/run_ldch.yaml"))
config_cdch = OctConfig.from_yaml(Path("config/run_cdch.yaml"))
config_lbp = OctConfig.from_yaml(Path("config/run_lbp.yaml"))
config_hog = OctConfig.from_yaml(Path("config/run_hog.yaml"))


@pytest.fixture(scope="module")
def benchmark_image_paths():
    valid_extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        [p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in valid_extensions]
    )
    assert len(paths) > 0, f"CRITICAL: No images found in {IMAGE_DIR}!"
    return paths


@pytest.fixture(scope="module")
def benchmark_images(benchmark_image_paths):
    images = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in benchmark_image_paths
    ]

    assert all(image is not None for image in images)

    return images


@pytest.fixture(scope="module")
def inferrer_ldch():
    return Inferrer(config_ldch, MODEL_PATH_LDCH)


@pytest.fixture(scope="module")
def inferrer_cdch():
    return Inferrer(config_cdch, MODEL_PATH_CDCH)


@pytest.fixture(scope="module")
def inferrer_lbp():
    return Inferrer(config_lbp, MODEL_PATH_LBP)


@pytest.fixture(scope="module")
def inferrer_hog():
    return Inferrer(config_hog, MODEL_PATH_HOG)


@pytest.fixture(scope="module")
def preprocessor():
    return PreprocessingRegistry.create_pipeline(config_ldch.preprocessing_pipeline)


@pytest.fixture(scope="module")
def preprocessed_images(benchmark_images, preprocessor):
    return [preprocessor.execute(image) for image in benchmark_images]


def test_preprocessing(benchmark, benchmark_images, preprocessor):
    def run():
        for image in benchmark_images:
            preprocessor.execute(image)

    benchmark(run)


def test_inference_array_ldch(
    benchmark,
    benchmark_images,
    inferrer_ldch,
):
    def run():
        for image in benchmark_images:
            inferrer_ldch.predict_array(image)

    benchmark(run)


def test_inference_array_cdch(
    benchmark,
    benchmark_images,
    inferrer_cdch,
):
    def run():
        for image in benchmark_images:
            inferrer_cdch.predict_array(image)

    benchmark(run)


def test_inference_array_lbp(
    benchmark,
    benchmark_images,
    inferrer_lbp,
):
    def run():
        for image in benchmark_images:
            inferrer_lbp.predict_array(image)

    benchmark(run)


def test_inference_array_hog(
    benchmark,
    benchmark_images,
    inferrer_hog,
):
    def run():
        for image in benchmark_images:
            inferrer_hog.predict_array(image)

    benchmark(run)

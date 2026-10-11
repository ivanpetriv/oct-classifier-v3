from pathlib import Path

import cv2

cv2.setNumThreads(1)
import pytest

from feature_extraction.extractors.cdch_feature_extractor import CdchFeaturesExtractor
from feature_extraction.extractors.hog_feature_extractor import HogFeaturesExtractor
from feature_extraction.extractors.lbp_feature_extractor import LbpFeaturesExtractor
from feature_extraction.extractors.ldch_feature_extractor import LdchFeaturesExtractor
from feature_extraction.extractors.ldch_py_feature_extractor import (
    LdchPyFeaturesExtractor,
)

IMAGE_DIR = Path(
    "/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli/test_preprocessed/"
)


@pytest.fixture(scope="module")
def benchmark_images():
    valid_extensions = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        [p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in valid_extensions]
    )[:100]

    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]

    assert all(image is not None for image in images)

    return images


def test_ldch_speed(benchmark, benchmark_images):
    extractor = LdchFeaturesExtractor(
        sections=128,
        layout="row",
        pixel_range=(1, 200),
        bins=64,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_ldch_py_speed(benchmark, benchmark_images):
    extractor = LdchPyFeaturesExtractor(
        sections=128,
        layout="row",
        pixel_range=(1, 200),
        bins=64,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_cdch_speed(benchmark, benchmark_images):
    extractor = CdchFeaturesExtractor(
        5,
        (0.44, 0.55),
        (30, 222),
        256,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_lbp_speed(benchmark, benchmark_images):
    extractor = LbpFeaturesExtractor(
        16,
        3,
        "nri_uniform",
        "l1",
        46,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


def test_hog_speed(benchmark, benchmark_images):
    extractor = HogFeaturesExtractor(
        10,
        (16, 16),
        (3, 3),
        "L2-Hys",
        False,
    )

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)


class NoOp:
    def extract(self, image):
        return None


def test_loop_baseline(benchmark, benchmark_images):
    extractor = NoOp()

    def run():
        for image in benchmark_images:
            extractor.extract(image)

    benchmark(run)

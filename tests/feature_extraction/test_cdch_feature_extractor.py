from pathlib import Path

from PIL import Image
import numpy as np
import pytest

from feature_extraction.extractors.cdch_feature_extractor import CdchFeaturesExtractor


SAMPLE_PATH = Path(__file__).parent.parent / "data" / "images" / "CNV-163081-45.jpeg"


@pytest.fixture
def sample_image() -> np.ndarray:
    """Fixture to load and provide the image array once per test session."""
    with Image.open(SAMPLE_PATH) as img:
        return np.array(img.convert("L"))


def test_cdch_benchmark(benchmark, sample_image: np.ndarray):
    extractor = CdchFeaturesExtractor(
        center=(0.5, 0.5), angle_step=1, pixel_range=(0, 254), bins=64
    )
    benchmark(extractor.extract, sample_image)

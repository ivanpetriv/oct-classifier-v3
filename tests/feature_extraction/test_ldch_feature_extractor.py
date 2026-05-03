from pathlib import Path

from PIL import Image
import numpy as np
import pytest

from feature_extraction.extractors.ldch_feature_extractor import LdchFeaturesExtractor


SAMPLE_PATH = Path(__file__).parent.parent / "data" / "images" / "CNV-163081-45.jpeg"


@pytest.fixture
def sample_image() -> np.ndarray:
    """Fixture to load and provide the image array once per test session."""
    with Image.open(SAMPLE_PATH) as img:
        return np.array(img.convert("L"))


def test_ldch_benchmark(benchmark, sample_image: np.ndarray):
    extractor = LdchFeaturesExtractor(
        sections=64, layout="row", pixel_range=(0, 254), bins=64
    )
    benchmark(extractor.extract, sample_image)

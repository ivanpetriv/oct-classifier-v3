from typing import Any

from feature_extraction.base_feature_extractor import BaseFeaturesExtractor
from feature_extraction.extractors.cdch_feature_extractor import CdchFeaturesExtractor
from feature_extraction.extractors.ldch_feature_extractor import LdchFeaturesExtractor
from feature_extraction.extractors.hog_feature_extractor import HogFeaturesExtractor
from feature_extraction.extractors.lbp_feature_extractor import LbpFeaturesExtractor
from feature_extraction.extractors.rc_ldch_feature_extractor import (
    RcLdchFeaturesExtractor,
)
from feature_extraction.extractors.rdch_feature_extractor import RdchFeaturesExtractor


class FeatureExtractorRegistry:
    """
    Factory that returns a feature extractor class based on its name as a string.
    """

    _registry: dict[str, type[BaseFeaturesExtractor]] = {
        "cdch": CdchFeaturesExtractor,
        "ldch": LdchFeaturesExtractor,
        "rdch": RdchFeaturesExtractor,
        "lbp": LbpFeaturesExtractor,
        "hog": HogFeaturesExtractor,
        "rcldch": RcLdchFeaturesExtractor,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseFeaturesExtractor:
        """
        Returns a feature extractor object based on its name as a string.
        :param name: Name of the feature extractor.
        :param kwargs: Arguments that the extractor accepts in its constructor.
        :return: Feature extractor object.
        """
        extractor_class = cls._registry.get(name.lower())

        if not extractor_class:
            raise ValueError(
                f"Unknown extractor '{name}'. Valid options: {list(cls._registry.keys())}"
            )

        return extractor_class(**kwargs)

import hashlib
import json
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from classification.registry import ClassifierRegistry
from config_settings.oct_config import OctConfig
from feature_extraction.registry import FeatureExtractorRegistry


def _get_feature_cache_dir(config: OctConfig) -> str | None:
    """
    Generates a unique cache path based on the feature extractor configurations.
    If configurations change or switch, a new subfolder is targeted automatically.
    """
    if not config.feature_extractor or not config.preprocessing_pipeline:
        return None

    extractor_info = {
        "feature_extractors": config.feature_extractor,
        "preprocessing": config.preprocessing_pipeline,
    }

    config_json = json.dumps(
        config.model_dump(include={"feature_extractor", "preprocessing_pipeline"}),
        sort_keys=True,
        default=str,
    )
    config_hash = hashlib.md5(config_json.encode()).hexdigest()

    cache_path = Path(config.paths.cache) / f"pipeline_cache_{config_hash}"
    return str(cache_path)


class SafePCA(PCA):
    def _cap_components(self, X):
        n_samples, n_features = X.shape

        limit = min(n_samples, n_features) - 1

        if self.n_components > limit:
            self.n_components = max(1, limit)

    def fit(self, X, y=None, **fit_params):
        self._cap_components(X)
        return super().fit(X, y, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        self._cap_components(X)
        return super().fit_transform(X, y, **fit_params)


def build_pipeline(
    config: OctConfig,
    image_preprocessing: bool,
    feature_extracting: bool,
    dimensionality_reduction: bool,
    classification: bool,
):
    pipeline: list[tuple[str, BaseEstimator]] = []

    if feature_extracting and config.feature_extractor:
        extracted_features = []

        for i, item in enumerate(config.feature_extractor):
            if isinstance(item, dict):
                extractor_name, extractor_params = next(iter(item.items()))
            else:
                extractor_name = item.active
                params_obj = item.params
                extractor_params = (
                    params_obj.model_dump(exclude_none=True)
                    if hasattr(params_obj, "model_dump")
                    else vars(params_obj)
                )

            extractor_instance = FeatureExtractorRegistry.create(
                extractor_name, **extractor_params
            )
            extracted_features.append(
                (f"extractor_{extractor_name}_{i}", extractor_instance)
            )

        if len(extracted_features) == 1:
            pipeline.append(extracted_features[0])
        else:
            pipeline.append(("feature_extractor", FeatureUnion(extracted_features)))

    pipeline.append(("scaler", StandardScaler()))

    if dimensionality_reduction and config.dimensionality_reduction:
        params_obj = config.dimensionality_reduction.params

        if hasattr(params_obj, "model_dump"):
            reduction_params = params_obj.model_dump(exclude_none=True)
        else:
            reduction_params = params_obj

        if config.dimensionality_reduction.active == "pca":
            dimensionality_reductor = SafePCA(copy=False, **reduction_params)
            pipeline.append(("dimensionality_reduction", dimensionality_reductor))

        elif config.dimensionality_reduction.active == "none":
            pipeline.append(("dimensionality_reduction", "passthrough"))

        else:
            raise ValueError(
                f"Unknown reducer: {config.dimensionality_reduction.active}"
            )

    if classification and config.classifier:
        classifier_params = vars(config.classifier.params).copy()

        # if "random_state" not in classifier_params:
        #     classifier_params["random_state"] = config.random_seed

        classifier = ClassifierRegistry.create(
            config.classifier.active, **classifier_params
        )
        pipeline.append(("classifier", classifier))

    pipeline_memory = _get_feature_cache_dir(config) if feature_extracting else None

    return Pipeline(pipeline, memory=None)

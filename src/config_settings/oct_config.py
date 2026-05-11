from typing import Literal, Any, Annotated, Union
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
import yaml

# --- Dataset and Global Settings ---


class Paths(BaseModel):
    normalized_dataset: Path
    original_dataset: Path
    cache: Path
    model_output: Path


class Dataset(BaseModel):
    class SamplingProportions(BaseModel):
        train: int = Field(..., gt=0)
        val: int = Field(..., gt=0)
        test: int = Field(..., gt=0)

    classes: list[str]
    samples_per_class: int = Field(..., gt=0)
    sampling_proportions: SamplingProportions


# --- Feature Extractor Components ---


class HOGConfig(BaseModel):
    class Params(BaseModel):
        orientations: int
        pixels_per_cell: tuple[int, int]
        cells_per_block: tuple[int, int]

    active: Literal["hog"]
    params: Params


class LBPConfig(BaseModel):
    class Params(BaseModel):
        n_points: int
        radius: int
        method: Literal["default", "ror", "uniform", "var"]

    active: Literal["lbp"]
    params: Params


class CDCHConfig(BaseModel):
    class Params(BaseModel):
        center: tuple[float, float]
        pixel_range: tuple[int, int]
        angle_step: int
        bins: int

    active: Literal["cdch"]
    params: Params


class DCHConfig(BaseModel):
    class Params(BaseModel):
        sections: int
        layout: Literal["row", "col"]
        pixel_range: tuple[int, int]
        bins: int

    active: Literal["dch"]
    params: Params


class RDCHConfig(BaseModel):
    class Params(BaseModel):
        rings: int
        center: tuple[float, float]
        pixel_range: tuple[int, int]
        bins: int

    active: Literal["rdch"]
    params: Params


class RCLDCHConfig(BaseModel):
    class Params(BaseModel):
        sections: int
        pixel_range: tuple[int, int]
        bins: int

    active: Literal["rcldch"]
    params: Params


# The discriminator routes validation to the correct subclass based on 'active'
FeatureExtractorConfig = Annotated[
    Union[HOGConfig, LBPConfig, CDCHConfig, DCHConfig, RDCHConfig, RCLDCHConfig],
    Field(discriminator="active"),
]

# --- Dimensionality Reduction Components ---


class PCAConfig(BaseModel):
    class Params(BaseModel):
        n_components: int | float
        whiten: bool = False

    active: Literal["pca"]
    params: Params


class NoneReducerConfig(BaseModel):
    active: Literal["none"]
    params: dict[str, Any] = Field(default_factory=dict)


ReducerConfig = Annotated[
    Union[PCAConfig, NoneReducerConfig], Field(discriminator="active")
]

# --- Classifier Components ---


class SVMConfig(BaseModel):
    class Params(BaseModel):
        kernel: Literal["linear", "poly", "rbf", "sigmoid"]
        C: float
        gamma: str | float
        probability: bool = False

    active: Literal["svm"]
    params: Params


class RandomForestConfig(BaseModel):
    class Params(BaseModel):
        n_estimators: int
        max_depth: int | None = None
        min_samples_split: int
        random_state: int | None = None

    active: Literal["random_forest"]
    params: Params


class KNNConfig(BaseModel):
    class Params(BaseModel):
        n_neighbors: int
        weights: Literal["uniform", "distance"]
        p: int

    active: Literal["knn"]
    params: Params


ClassifierConfig = Annotated[
    Union[SVMConfig, RandomForestConfig, KNNConfig], Field(discriminator="active")
]

# --- Main Configuration Model ---


class OctConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: Paths
    random_seed: int

    dataset: Dataset

    preprocessing_pipeline: list[dict[str, dict[str, Any]]]

    feature_extractor: FeatureExtractorConfig
    dimensionality_reduction: ReducerConfig
    classifier: ClassifierConfig

    @classmethod
    def from_yaml(cls, filepath: Path | str) -> "OctConfig":
        """Loads and validates the configuration from a YAML file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as file:
            config_dict = yaml.safe_load(file)

        return cls.model_validate(config_dict)

    def print_summary(self):
        """Prints a human-readable JSON representation of the active config."""
        print(self.model_dump_json(indent=4, exclude_unset=True))

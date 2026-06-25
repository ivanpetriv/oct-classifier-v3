from typing import Literal, Any, Annotated, Union
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# Global / Dataset
# ============================================================


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


# ============================================================
# Feature Extractors
# ============================================================


class HOGConfig(BaseModel):
    class Params(BaseModel):
        orientations: int

        pixels_per_cell: tuple[int, int]

        cells_per_block: tuple[int, int]

        transform_sqrt: bool = False

        block_norm: Literal[
            "L1",
            "L1-sqrt",
            "L2",
            "L2-Hys",
        ] = "L2-Hys"

    active: Literal["hog"]

    params: Params


class LBPConfig(BaseModel):
    class Params(BaseModel):
        n_points: int

        radius: int

        method: Literal[
            "default",
            "ror",
            "uniform",
            "nri_uniform",
            "var",
        ]

        norm: Literal[
            "l1",
            "l2",
        ]

        padding_threshold: int

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


class LDCHConfig(BaseModel):
    class Params(BaseModel):
        sections: int

        layout: Literal["row", "col"]

        pixel_range: tuple[int, int]

        bins: int

    active: Literal["ldch"]

    params: Params


class RDCHConfig(BaseModel):
    class Params(BaseModel):
        ring_step: int

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


FeatureExtractorConfig = Annotated[
    Union[
        HOGConfig,
        LBPConfig,
        CDCHConfig,
        LDCHConfig,
        RDCHConfig,
        RCLDCHConfig,
    ],
    Field(discriminator="active"),
]


# ============================================================
# Dimensionality Reduction
# ============================================================


class PCAConfig(BaseModel):
    class Params(BaseModel):
        n_components: int

        whiten: bool

        svd_solver: Literal[
            "auto",
            "full",
            "covariance_eigh",
            "arpack",
            "randomized",
        ]

        iterated_power: int | Literal["auto"] = "auto"

        power_iteration_normalizer: Literal[
            "auto",
            "QR",
            "LU",
            "none",
        ] = "auto"

        random_state: int | None = None

    active: Literal["pca"]

    params: Params


class NoneReducerConfig(BaseModel):
    active: Literal["none"]

    params: dict[str, Any] = Field(default_factory=dict)


ReducerConfig = Annotated[
    Union[
        PCAConfig,
        NoneReducerConfig,
    ],
    Field(discriminator="active"),
]


# ============================================================
# Classifiers
# ============================================================


class SVMConfig(BaseModel):
    class LinearParams(BaseModel):
        kernel: Literal["linear"]
        C: float
        shrinking: bool
        class_weight: str | None

    class RbfParams(BaseModel):
        kernel: Literal["rbf"]
        C: float
        gamma: str | float
        shrinking: bool
        class_weight: str | None

    class PolyParams(BaseModel):
        kernel: Literal["poly"]
        C: float
        gamma: str | float
        degree: int
        shrinking: bool
        class_weight: str | None

    active: Literal["svm"]
    params: Union[LinearParams, RbfParams, PolyParams] = Field(
        ..., discriminator="kernel"
    )


class RandomForestConfig(BaseModel):
    class Params(BaseModel):
        n_estimators: int

        max_depth: int | None = None

        min_samples_split: int

        min_samples_leaf: int = 1

        max_features: (
            Literal[
                "sqrt",
                "log2",
            ]
            | None
        ) = "sqrt"

        bootstrap: bool = True

        max_samples: float | None = None

        criterion: Literal[
            "gini",
            "entropy",
            "log_loss",
        ] = "gini"

        class_weight: (
            Literal[
                "balanced",
                "balanced_subsample",
            ]
            | None
        ) = None

        random_state: int | None = None

        n_jobs: int | None = None

    active: Literal["random_forest"]

    params: Params


class KNNConfig(BaseModel):
    class Params(BaseModel):
        n_neighbors: int

        weights: Literal[
            "uniform",
            "distance",
        ]

        p: int

        metric: Literal[
            "minkowski",
            "cosine",
        ] = "minkowski"

        algorithm: Literal[
            "auto",
            "ball_tree",
            "kd_tree",
            "brute",
        ] = "auto"

        leaf_size: int = 30

    active: Literal["knn"]

    params: Params


ClassifierConfig = Annotated[
    Union[
        SVMConfig,
        RandomForestConfig,
        KNNConfig,
    ],
    Field(discriminator="active"),
]


# ============================================================
# Main Config
# ============================================================


class OctConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    paths: Paths

    random_seed: int

    dataset: Dataset

    preprocessing_pipeline: list[dict[str, dict[str, Any]]]

    feature_extractor: FeatureExtractorConfig

    dimensionality_reduction: ReducerConfig

    classifier: ClassifierConfig

    @classmethod
    def from_yaml(
        cls,
        filepath: Path | str,
    ) -> "OctConfig":
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as file:
            config_dict = yaml.safe_load(file)

        return cls.model_validate(config_dict)

    def print_summary(self):
        print(
            self.model_dump_json(
                indent=4,
                exclude_unset=True,
            )
        )

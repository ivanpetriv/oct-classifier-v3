from typing import Any

from pydantic import BaseModel

from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.preprocessing_step import PreprocessingStep
from preprocessing.steps.bilateral_denoiser import BilateralDenoiser
from preprocessing.steps.clahe_filter import ClaheFilter
from preprocessing.steps.corner_artifact_remover import CornerArtifactRemover
from preprocessing.steps.cropper import Cropper
from preprocessing.steps.dwt_denoiser import DwtDenoiser
from preprocessing.steps.resizer import Resizer
from preprocessing.steps.rpe_aligner import RpeAligner
from preprocessing.steps.rpe_roi_extractor import RpeRoiExtractor


class PreprocessingRegistry:
    _registry: dict[str, type[PreprocessingStep]] = {
        "bilateral_denoiser": BilateralDenoiser,
        "dwt_denoiser": DwtDenoiser,
        "clahe_filter": ClaheFilter,
        "corner_artifact_remover": CornerArtifactRemover,
        "rpe_aligner": RpeAligner,
        "rpe_roi_aligner": RpeRoiExtractor,
        "resizer": Resizer,
        "cropper": Cropper,
    }

    @classmethod
    def create_pipeline(
        cls, pipeline_config: list[dict[str, dict[str, Any]]]
    ) -> PreprocessingPipeline:
        """
        Factory method to create a preprocessing pipeline based on an ordered
        list of step configurations.
        """
        pipeline_steps = []

        for step in pipeline_config:
            for step_name, params in step.items():
                if step_name not in cls._registry:
                    raise ValueError(f"Unknown preprocessing step: {step_name}")

                if isinstance(params, BaseModel):
                    kwargs = params.model_dump()
                else:
                    kwargs = params

                pipeline_steps.append(cls._registry[step_name](**kwargs))

        return PreprocessingPipeline(pipeline_steps)

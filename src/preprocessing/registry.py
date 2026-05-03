from typing import Any

from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.preprocessing_step import PreprocessingStep
from preprocessing.steps.bilateral_denoiser import BilateralDenoiser
from preprocessing.steps.clahe_filter import ClaheFilter
from preprocessing.steps.corner_artifact_remover import CornerArtifactRemover
from preprocessing.steps.dwt_denoiser import DwtDenoiser
from preprocessing.steps.rpe_aligner import RpeAligner


class PreprocessingRegistry:
    _registry: dict[str, type[PreprocessingStep]] = {
        "bilateral_denoiser": BilateralDenoiser,
        "dwt_denoiser": DwtDenoiser,
        "clahe_filter": ClaheFilter,
        "corner_artifact_remover": CornerArtifactRemover,
        "rpe_aligner": RpeAligner,
    }

    @classmethod
    def create_pipeline(
        cls, execution_order: list[str], steps_config: dict[str, dict[str, Any]]
    ) -> PreprocessingPipeline:
        """
        Factory method to create a preprocessing pipeline based on the execution order
        and step parameters.
        """
        pipeline_steps: list[PreprocessingStep] = []

        for step_name in execution_order:
            if step_name not in steps_config:
                raise KeyError(
                    f"Step '{step_name}' is in execution_order but missing from parameters."
                )

            if step_name not in cls._registry:
                raise ValueError(
                    f"Preprocessing step '{step_name}' is not registered in the Factory."
                )

            params = steps_config[step_name]
            pipeline_steps.append(cls._registry[step_name](**params))

        return PreprocessingPipeline(pipeline_steps)

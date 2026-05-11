from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from config_settings.oct_config import OctConfig
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.registry import PreprocessingRegistry


class PreprocessingStepsVisualizer:
    def __init__(self, pipeline: PreprocessingPipeline):
        self.pipeline = pipeline

    def visualize(self, image: np.ndarray, save_path: str | None = None):
        image = image.copy()

        image_stages: dict[str, np.ndarray] = {}
        image_stages["Orignial"] = image.copy()

        def capture_stage(step, index, image):
            image_stages[step.__class__.__name__] = image.copy()

        pipeline.add_hook("after_step", capture_stage)

        try:
            self.pipeline.execute(image)

            n_stages = len(image_stages)
            _, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))

            for ax, (title, display_array) in zip(axes, image_stages.items()):
                if display_array.dtype.kind in {"f", "c"}:
                    display_array = np.clip(display_array, 0.0, 1.0)
                    display_array = (display_array * 255.0).astype(np.uint8)

                if len(display_array.shape) == 2:
                    ax.imshow(display_array, cmap="gray")
                else:
                    ax.imshow(display_array)

                ax.set_title(title, fontsize=12, pad=10)
                ax.axis("off")

            plt.tight_layout()
            plt.show()
        finally:
            pipeline.remove_hook("after_step")


if __name__ == "__main__":
    config = OctConfig.from_yaml("config/run.yaml")

    pipeline = PreprocessingRegistry.create_pipeline(config.preprocessing_pipeline)

    path = r"/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli/test/CNV/CNV-103044-104.jpeg"
    # path = r"/run/media/ivan/01DCA2451BEF4120/LPNU/University/masters/Thesis/Ravelli/test/CNV/CNV-163081-32.jpeg"

    visualizer = PreprocessingStepsVisualizer(pipeline)
    with Image.open(path) as image_obj:
        visualizer.visualize(np.array(image_obj))

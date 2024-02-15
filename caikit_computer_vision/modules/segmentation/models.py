# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for image segmentation models.
"""
# Standard
from typing import Union
import os

# Third Party
from torchvision.transforms.v2.functional import pil_to_tensor
import torch

# First Party
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.vision import data_model as caikit_dm
from caikit.interfaces.vision.data_model.backends import image_pil_backend
import alog

# Local
from ...data_model import BoundingBox, ImageSegmentationResult, ObjectSegment
from ...data_model.tasks import ImageSegmentationTask
from .image_processing import maskrcnn_postprocess

log = alog.use_channel("IMAGE_SEGMENTATION")
error = error_handler.get(log)


@module(
    id="8vq41ck8-mbvt-s2u3-ym3h-e8f4p35xoluz",
    name="Vision Transformers based Segmentation",
    version="0.1.0",
    task=ImageSegmentationTask,
)
class ViTSegmenter(ModuleBase):
    _SEGMENTATION_ARTIFACTS_CONFIG_KEY = "segmentation_artifacts"

    def __init__(
        self,
        model_name: str,
        segmentation_model: str,
    ) -> "ViTSegmenter":
        """Initialize a caikit-wrapped segmentation model."""
        super().__init__()
        self.model_name = model_name
        self.post_processor = maskrcnn_postprocess
        self.segmentation_model = segmentation_model

    @classmethod
    def load(cls, model_file: Union[str, "ModuleConfig"]) -> "ViTSegmenter":
        """Load an instance of this class.

        Args
            model_file: Union[str, "ModuleConfig"]
                Path to model to be loaded from disk or raw ModuleConfig to be
                leveraged.

        Returns:
            ViTSegmenter
                Instance of this class.
        """
        config = ModuleConfig.load(model_file)
        segmentation_path = os.path.join(
            config.model_path, config[cls._SEGMENTATION_ARTIFACTS_CONFIG_KEY]
        )
        error.dir_check("<CCV61312083E>", segmentation_path)
        return cls.bootstrap(
            model_file=os.path.join(
                segmentation_path, os.path.basename(config.model_name)
            )
        )

    def save(
        self,
        model_path: str,
    ):
        """Save the in-memory model to the given path.

        Args:
            model_path: str
                Path that we want to export the segmentation model to.
        """
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        segmentation_rel_path, segmentation_abs_path = saver.add_dir(model_path)
        with saver:
            saver.update_config(
                {
                    self._SEGMENTATION_ARTIFACTS_CONFIG_KEY: segmentation_rel_path,
                    "model_name": self.model_name,
                }
            )
            torch.jit.save(
                self.segmentation_model,
                os.path.join(segmentation_abs_path, os.path.basename(self.model_name)),
            )

    @classmethod
    def bootstrap(
        cls,
        model_file: str,
    ) -> "ViTSegmenter":
        """Create an instance of this class from a segmentation model.

        Args:
            model_file: str
                Pretrained filepath of the segmentation model to be loaded.

        Returns:
            ViTSegmenter
                An instance of this class.
        """
        segmentation_model = torch.jit.load(model_file)
        return cls(model_file, segmentation_model)

    def run(
        self, inputs: image_pil_backend.PIL_SOURCE_TYPES, threshold: float = 0.2
    ) -> ImageSegmentationResult:
        """Run inference on a single image.

        Args:
            inputs: PIL_SOURCE_TYPES
                Image to be processed; must be either a caikit image, or a type that
                can be coerced into one. Currently this includes the following types:
                   [PIL.Image.Image, pathlib.PosixPath, str, numpy.ndarray, bytes]
            threshold: float
                Threshold in range (0,1] to be used for filtering bounding box predictions.
                Default 0.2

        Returns:
            ImageSegmentationResult
                Bounding box and mask predictions for this image.
        """

        # Coerce to a caikit Image
        if not isinstance(inputs, caikit_dm.Image):
            inputs = caikit_dm.Image(inputs)

        # View the caikit image as a tensor & run the segmentation model
        image_tensor = pil_to_tensor(inputs.as_pil())
        outputs = self.segmentation_model(image_tensor)

        num_objects = len(outputs[0])
        log.debug("Detected [%d] objects", num_objects)

        if num_objects:
            results_dict = {
                key: value.cpu().detach().numpy().tolist()
                for key, value in zip(
                    ["BOXES", "CLASSES", "MASKS", "SCORES", "IMAGES_SIZE"], outputs
                )
            }
            results = self.post_processor(
                results_dict, "testuuid", threshold=threshold
            )["annotations"]
        else:
            results = {}

        # Convert the results into a Caikit DM object
        segmented_objects = [
            ObjectSegment(
                score=results[idx]["attributes"]["score"],
                category_id=results[idx]["category_id"],
                bbox=BoundingBox(*results[idx]["bbox"]),
                polygon=results[idx]["segmentation"],
                area=results[idx]["area"],
            )
            for idx in range(num_objects)
        ]

        return ImageSegmentationResult(object_segments=segmented_objects)

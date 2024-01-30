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
"""Module for transformers object detector transformer models.
"""
# Standard
from typing import Union, get_args
import os

# Third Party
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import numpy as np

# First Party
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.vision import data_model as caikit_dm
from caikit.interfaces.vision.data_model.backends import image_pil_backend
import alog

# Local
from ...data_model import (
    BoundingBox,
    DetectedObject,
    FlatImage,
    ObjectDetectionResult,
    ObjectDetectionTrainSet,
)
from ...data_model.tasks import ObjectDetectionTask

log = alog.use_channel("TRANSFORMERS_DETECT")
error = error_handler.get(log)

# First Party
from caikit.interfaces.vision.data_model.backends import image_pil_backend

SUPPORTED_INFERENCE_TYPES = Union[
    tuple(list(get_args(image_pil_backend.PIL_SOURCE_TYPES)) + [FlatImage])
]


@module(
    id="28dc918a-4e18-41c0-22a3-aa9c3c5caa11",
    name="Transformers AutoModelForObjectDetection",
    version="0.1.0",
    task=ObjectDetectionTask,
)
class TransformersObjectDetector(ModuleBase):
    _IMPROC_ARTIFACTS_CONFIG_KEY = "image_processor_artifacts"
    _DETECTOR_ARTIFACTS_CONFIG_KEY = "object_detector_artifacts"

    def __init__(
        self,
        model_name: str,
        image_processor: "BaseImageProcessor",
        detector_model: "PreTrainedModel",
    ) -> "TransformersObjectDetector":
        """Initialize a caikit-wrapped object detector."""
        super().__init__()
        self.model_name = model_name
        self.image_processor = image_processor
        self.detector_model = detector_model

    @classmethod
    def load(
        cls, model_path: Union[str, "ModuleConfig"]
    ) -> "TransformersObjectDetector":
        """Load an instance of this class.

        Args
            model_path: Union[str, "ModuleConfig"]
                Path to model to be loaded from disk or raw ModuleConfig to be
                leveraged.

        Returns:
            TransformersObjectDetector
                Instance of this class.
        """
        config = ModuleConfig.load(model_path)
        improc_path = os.path.join(
            config.model_path, config[cls._IMPROC_ARTIFACTS_CONFIG_KEY]
        )
        detector_path = os.path.join(
            config.model_path, config[cls._DETECTOR_ARTIFACTS_CONFIG_KEY]
        )
        error.dir_check("<CCV18883415E>", improc_path)
        error.dir_check("<CCV19293413E>", detector_path)
        return cls.bootstrap(
            model_name=detector_path,
            improc_name=improc_path,
        )

    def save(
        self,
        model_path: str,
        image_proc_dirname: str = "image_processor",
        detector_dirname: str = "object_detector",
    ):
        """Save the in-memory model to the given path.

        Args:
            model_path: str
                Path that we want to export our object detector to.
            image_proc_dirname: str
                Subdirectory to which we want to save the image processor.
                Default: image_processor
            detector_dirname: str
                Subdirectory to which we want to save the detector.
                Default: object_detector
        """
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        improc_rel_path, improc_abs_path = saver.add_dir(image_proc_dirname)
        det_rel_path, det_abs_path = saver.add_dir(detector_dirname)
        with saver:
            saver.update_config(
                {
                    self._IMPROC_ARTIFACTS_CONFIG_KEY: improc_rel_path,
                    self._DETECTOR_ARTIFACTS_CONFIG_KEY: det_rel_path,
                    "model_name": self.model_name,
                }
            )
            self.image_processor.save_pretrained(improc_abs_path)
            self.detector_model.save_pretrained(det_abs_path)

    @classmethod
    def bootstrap(
        cls,
        model_name: str,
        improc_name: str = None,
    ) -> "TransformersObjectDetector":
        """Create an instance of this class from an image processor or detector model.
        # TODO: Add env var override & deny download by default.

        Args:
            model_name: str
                Name of the detector to be loaded.
            improc_name: str
                Name of the image processor to be loaded; if none is provided, the name
                for the detector will be used.

        Returns:
            TransformersObjectDetector
                An instance of this class.
        """
        if improc_name is None:
            improc_name = model_name
        image_processor = AutoImageProcessor.from_pretrained(improc_name)
        detector_model = AutoModelForObjectDetection.from_pretrained(model_name)
        return cls(model_name, image_processor, detector_model)

    def run(
        self, inputs: SUPPORTED_INFERENCE_TYPES, threshold: float = 0.5
    ) -> ObjectDetectionResult:
        """Run inference on a single image.

        Args:
            inputs: PIL_SOURCE_TYPES
                Image to be processed; must be either a caikit image, or a type that
                can be coerced into one. Currently this includes the following types:
                   [PIL.Image.Image, pathlib.PosixPath, str, numpy.ndarray, bytes]
            threshold: float
                Threshold in range (0,1] to be used for filtering bounding box predictions.
                Default 0.5

        Returns:
            ObjectDetectionResult
                Bounding box predictions for this image.
        """
        # If we have a FlatImage, we currently have to handle it separately, since we have
        # not added coersion support for it into the caikit image yet. To do this, we just
        # convert it into a numpy array, then rely on the numpy coersion logic.
        # TODO: Port to caikit and initialize the DM directly with the flat image.
        if isinstance(inputs, FlatImage):
            inputs = inputs.to_numpy()

        # Coerce to a caikit Image
        if not isinstance(inputs, caikit_dm.Image):
            inputs = caikit_dm.Image(inputs)

        # View the caikit image as a PIL image for processing & run the detector
        detector_inputs = self.image_processor(
            images=inputs.as_pil(), return_tensors="pt"
        )
        outputs = self.detector_model(**detector_inputs)

        # .run() operates over a single image, so we only get one pred dict back
        result = self.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(inputs.rows, inputs.columns)]
        )[0]

        # Convert the result dictionary into a Caikit DM object
        num_objects = result["scores"].shape[0]
        log.debug("Detected [%d] objects", num_objects)
        detected_objects = [
            DetectedObject(
                score=result["scores"][idx].item(),
                label=str(result["labels"][idx].item()),
                box=BoundingBox(
                    *[int(coord) for coord in result["boxes"][idx].tolist()]
                ),
            )
            for idx in range(num_objects)
        ]

        return ObjectDetectionResult(detected_objects=detected_objects)

    @classmethod
    def train(
        cls,
        model_path: str,
        train_data: ObjectDetectionTrainSet,  # pylint: disable=W0613
        num_epochs: int,
        learning_rate: float,
    ) -> "TransformersObjectDetector":
        """Stub for train for demonstration purposes through runtime; currently
        this is essentially an alias to load with some passable params that do
        nothing, just so we can demonstrate the interface.
        """
        log.debug("STUB - Training detector")
        return cls.load(model_path)

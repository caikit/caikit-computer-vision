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
"""
This module holds the Task definitions for all common vision tasks
"""

# Standard
from typing import Union

# First Party
from caikit.core import TaskBase, task

# Local
from .flat_image import FlatImage
from .image_classification import ImageClassificationResult
from .image_segmentation import ImageSegmentationResult
from .object_detection import ObjectDetectionResult
from .text_to_image import CaptionedImage


# TODO - add support for image DM primitives
@task(
    required_parameters={"inputs": Union[bytes, str, FlatImage]},
    output_type=ObjectDetectionResult,
)
class ObjectDetectionTask(TaskBase):
    """The Object Detection Task is responsible for taking an input image
    and producing 0 or more detected objects, which typically include labels
    and confidence scores.
    """


@task(
    required_parameters={"inputs": Union[bytes, str, FlatImage]},
    output_type=ImageClassificationResult,
)
class ImageClassificationTask(TaskBase):
    """The image classification task is responsible for taking an input image
    and producing an iterable of objects containing class names and typically
    confidence scores.
    """


@task(
    required_parameters={"inputs": Union[bytes, str, FlatImage]},
    output_type=ImageSegmentationResult,
)
class ImageSegmentationTask(TaskBase):
    """The image classification task is responsible for taking an input image
    and producing a pixel mask with optional class names and confidence scores.
    Note that at the moment, this task encapsulates all segmentation types,
    I.e., instance, object, semantic, etc...
    """


@task(
    required_parameters={"inputs": str},
    output_type=CaptionedImage,
)
class TextToImageTask(TaskBase):
    """The text to image task is responsible for taking an input text prompt, along with
    other optional image generation parameters, e.g., image height and width,
    and generating an image.
    """

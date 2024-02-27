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
# Standard
import os

# Third Party
import pytest

# Local
from caikit_computer_vision.modules.object_detection import TransformersObjectDetector
from caikit_computer_vision.modules.segmentation import ViTSegmenter

### Constants used in fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__))
TINY_MODELS_DIR = os.path.join(FIXTURES_DIR, "tiny_models")
TRANSFORMER_OBJ_DETECT_MODEL = os.path.join(TINY_MODELS_DIR, "YolosForObjectDetection")
SEMANTIC_SEGMENTATION_MODEL_DIR = os.path.join(TINY_MODELS_DIR, "ImageSegmentation")
SEGMENTATION_MODEL_CKPT = os.path.join(SEMANTIC_SEGMENTATION_MODEL_DIR, "model.pt")


@pytest.fixture
def detector_transformer_dummy_model():
    """Bootstrap a detector transformer dummy model [yolos]."""
    return TransformersObjectDetector.bootstrap(TRANSFORMER_OBJ_DETECT_MODEL)


@pytest.fixture
def segmentation_dummy_model():
    """Load torch scripted model weights for ViT Segmentation"""
    return ViTSegmenter.bootstrap(SEGMENTATION_MODEL_CKPT)

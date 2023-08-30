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
import tempfile

# Third Party
import numpy as np
import pytest

# Local
from caikit_computer_vision.data_model import ObjectDetectionResult
from caikit_computer_vision.modules.object_detection import TransformersObjectDetector
from tests.fixtures import (
    TRANSFORMER_OBJ_DETECT_MODEL,
    detector_transformer_dummy_model,
)


## Tests #######################################################################
def test_bootstrap_and_run():
    """Ensure that we can bootstrap a model & run inference on it."""
    model = TransformersObjectDetector.bootstrap(TRANSFORMER_OBJ_DETECT_MODEL)
    assert isinstance(model, TransformersObjectDetector)
    preds = model.run(np.ones((3, 3, 3), dtype=np.uint8))
    assert isinstance(preds, ObjectDetectionResult)


def test_save_model_can_load():
    """Ensure that we can save a model and reload it."""
    model = TransformersObjectDetector.bootstrap(TRANSFORMER_OBJ_DETECT_MODEL)
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        del model
        new_model = TransformersObjectDetector.load(model_dir)
        preds = new_model.run(np.ones((3, 3, 3), dtype=np.uint8))
        assert isinstance(preds, ObjectDetectionResult)

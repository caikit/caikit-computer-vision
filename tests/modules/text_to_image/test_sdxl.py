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
from tempfile import TemporaryDirectory
import os

# Local
from caikit_computer_vision.modules.text_to_image import SDXLStub
import caikit_computer_vision


def test_sdxl_stub():
    """Ensure that the stubs for load / save / run work as expected."""
    # Make sure we can bootstrap a model
    model = SDXLStub.bootstrap("foo")
    assert isinstance(model, SDXLStub)

    # Make sure we can run a fake inference on it
    pred = model.run("This is a prompt", height=500, width=550)
    pil_img = pred.output.as_pil()
    assert pil_img.width == 550
    assert pil_img.height == 500

    # Make sure we can save the model
    model_dirname = "my_model"
    with TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, model_dirname)
        model.save(model_path)
        reloaded_model = model.load(model_path)

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

# Third Party
import pytest
import torch

# Local
from caikit_computer_vision.data_model import CaptionedImage
from caikit_computer_vision.modules.text_to_image import SDXL
from tests.fixtures import sdxl_dummy_model

TEST_MODEL_NAME = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


def test_bootstrap_sdxl():
    """Ensure we can bootstrap a model and get an SDXL instance back."""
    sdxl_module = SDXL.bootstrap(TEST_MODEL_NAME)
    assert isinstance(sdxl_module, SDXL)


def test_save_and_reload_sdxl(sdxl_dummy_model):
    """Ensure we can save an SDXL model."""
    export_subpath = "sdxl_test_model"
    with TemporaryDirectory() as dirname:
        export_path = os.path.join(dirname, export_subpath)
        sdxl_dummy_model.save(export_path)
        # Ensure that the model exists and can be reloaded
        assert os.path.exists(export_path)
        reloaded_module = SDXL.load(export_path)
        assert isinstance(reloaded_module, SDXL)


def test_run_sdxl_required_params(sdxl_dummy_model):
    """Ensure that we can run inference on an SDXL model."""
    prompt = "a picture of a dog sitting in a grassy field"
    res = sdxl_dummy_model.run(prompt)
    assert isinstance(res, CaptionedImage)
    # We should get a (512, 512, 3) image out
    assert res.output.columns == 512
    assert res.output.rows == 512
    assert res.output.channels == 3
    assert res.caption == prompt


def test_run_sdxl_with_varying_height_and_width(sdxl_dummy_model):
    """Ensure that we can run inference on an SDXL model with optional params."""
    res = sdxl_dummy_model.run(
        "a picture of a dog sitting in a grassy field", height=256, width=264
    )
    assert isinstance(res, CaptionedImage)
    assert res.output.columns == 264
    assert res.output.rows == 256
    assert res.output.channels == 3


@pytest.mark.parametrize(
    "inference_args",
    ({"height": -8}, {"width": -8}, {"num_steps": -1}),
)
def test_run_sdxl_with_invalid_args(sdxl_dummy_model, inference_args):
    """Test edge cases that are invalid for inference."""
    with pytest.raises(ValueError):
        sdxl_dummy_model.run(inputs="this is a test prompt", **inference_args)


def test_height_and_width_are_rounded_to_nearest_multiple_of_eight(sdxl_dummy_model):
    """Ensure that we can don't fail if the dims are not multiples of eight."""
    res = sdxl_dummy_model.run(
        "a picture of a dog sitting in a grassy field", height=253, width=259
    )
    assert isinstance(res, CaptionedImage)
    assert res.output.columns == 256
    assert res.output.rows == 256
    assert res.output.channels == 3


@pytest.mark.parametrize(
    "device_arg",
    (
        "cpu",
        "cuda",
    ),
)
def test_device_initialization_bootstrap(device_arg):
    """Test model device placement; note that load wraps bootstrap, so this tests load also."""
    if device_arg != "cuda" or (device_arg == "cuda" and torch.cuda.is_available()):
        sdxl_module = SDXL.bootstrap(TEST_MODEL_NAME, device=device_arg)
        assert sdxl_module.pipeline.device.type == device_arg


def test_infer_device__bootstrap():
    """Test that if no device is specifed, we load on cuda device if available, else cpu."""
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    sdxl_module = SDXL.bootstrap(TEST_MODEL_NAME)
    assert sdxl_module.pipeline.device.type == expected_device


@pytest.mark.parametrize(
    "image_format",
    (
        "png",
        "jpeg",
        "webp",
    ),
)
def test_image_serialization_format(sdxl_dummy_model, image_format):
    """Ensure that we can control the format of the encapsulated image at inference time."""
    res = sdxl_dummy_model.run(
        "a picture of a dog sitting in a grassy field",
        height=256,
        width=256,
        image_format=image_format,
    )
    # Ensure that if we go to proto & back, the image format is preserved
    reconstructed_result = CaptionedImage.from_proto(res.to_proto())
    rebuilt_pil_image = reconstructed_result.output.as_pil()
    rebuilt_pil_image.format.lower() == image_format


def test_image_serialization_bad_format(sdxl_dummy_model):
    """Ensure that passing a bad image format raises an error at inference time."""
    with pytest.raises(KeyError):
        sdxl_dummy_model.run(
            "a picture of a dog sitting in a grassy field",
            height=256,
            width=256,
            image_format="garbage",
        )

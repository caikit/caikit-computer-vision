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
"""Module for text to image via SDXL.
"""
# Standard
from typing import Any, Dict, Optional, Union
import os

# Third Party
from diffusers import AutoPipelineForText2Image
from PIL.Image import SAVE, init
import torch

# First Party
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.vision import data_model as caikit_dm
import alog

# Local
from ...data_model import CaptionedImage
from ...data_model.tasks import TextToImageTask

log = alog.use_channel("SDXL")
error = error_handler.get(log)


@module(
    id="28aa777c-1b13-21b0-11b3-bb9c3b0cbb56",
    name="Text to Image via SDXL",
    version="0.1.0",
    task=TextToImageTask,
)
class SDXL(ModuleBase):
    _DETECT_DEVICE = "__DETECT__"
    _SDXL_PIPELINE_CONFIG_KEY = "sdxl_model"

    def __init__(
        self,
        model_name: str,
        pipeline: Any,  # TODO: test this with SD, but usually a StableDiffusionXLPipeline
    ) -> "SDXL":
        """Initialize a wrapper around an SDXL text to image pipeline.

        Args:
            model_name: str
                Name of the model being initialized.
            pipeline: Any
                Initialized pipeline to be wrapped.
        """
        super().__init__()
        self.model_name = model_name
        self.pipeline = pipeline

    @classmethod
    def load(
        cls, model_path: Union[str, "ModuleConfig"], device: str = _DETECT_DEVICE
    ) -> "SDXL":
        """Loads an instance of this class from an saved caikit module.

        Args:
            model_path: Union[str, "ModuleConfig"]
                Path to the caikit model to be loaded.
            device: str
                The device to load the model onto (follows device name convention used by pytorch).

        Returns:
            SDXL
                An instance of this class wrapping the model indicated by model_name on the correct
                device.
        """
        config = ModuleConfig.load(model_path)
        pipeline_path = os.path.join(
            config.model_path, config[SDXL._SDXL_PIPELINE_CONFIG_KEY]
        )
        return cls.bootstrap(pipeline_path, device)

    @classmethod
    def bootstrap(
        cls, model_name: str, device: str = _DETECT_DEVICE, **pipeline_kwargs
    ) -> "SDXL":
        """Creates an instance of this class from an external model, i.e., local or
        on HuggingFaceHub.

        Args:
            model_name: str
                The model that we would like to bootstrap.
            device: str
                The device to load the model onto (follows device name convention used by pytorch).
            **pipeline_kwargs
                Additional kwargs to be passed to the pipeline creation, i.e.,
                AutoPipelineForText2Image.from_pretrained, e.g., revision, etc.

        Returns:
            SDXL
                An instance of this class wrapping the model indicated by model_name on the correct
                device.
        """
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_name, **pipeline_kwargs
        )
        device = cls._get_device(device)
        log.warning(f"Loading Text to image pipline on device [{device}]")
        pipeline = pipeline.to(device)
        return cls(model_name, pipeline)

    def save(self, model_path: str):
        """Saves the pipeline model.

        Args:
            model_path: str
                Path to the model we would like to save.
        """
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with saver:
            model_rel_path, model_abs_path = saver.add_dir(
                self._SDXL_PIPELINE_CONFIG_KEY
            )
            saver.update_config(
                {
                    "model_name": self.model_name,
                    self._SDXL_PIPELINE_CONFIG_KEY: model_rel_path,
                }
            )
            self.pipeline.save_pretrained(model_abs_path)

    def run(
        self,
        inputs: str,
        height: int = 512,
        width: int = 512,
        num_steps: int = 1,
        guidance_scale: float = 0.0,
        negative_prompt: Optional[str] = None,
        image_format: str = "png",
    ) -> CaptionedImage:
        """Generates an image matching the provided height and width.

        NOTE: We currently expose guidance scale / negative prompt as args, but they should be
        unset for SDXL turbo, since the model does not leverage them.

        Args:
            inputs: str
                Text prompt to be used for the generation.
            height: int
                Height of the image to be generated.
            width: int
                Width of the image to be generated.
            num_steps: int
                Number of steps to be used in image generation.
            guidance_scale: float
                Guidance scale to be used; set this to 0.0 for SDXL turbo.
            negative_prompt: Optional[str]
                Negative prompt to be used; leave this unset for SDXL turbo.
            image_format: str
                Format to be used for the underlying PIL object, e.g., at serialization
                time, etc.

        Returns:
            TextToImageResult
                Object encapsulating the generated image.
        """
        error.value_check(
            "<CCV81444491E>",
            height > 0 and width > 0,
            "Height & width must be positive values",
        )
        error.value_check(
            "<CCV14111912E>",
            num_steps > 0,
            "Number of steps must be a positive value",
        )
        SDXL._validate_image_format(image_format)
        dims_dict = self._force_to_nearest_multiples_of_eight(
            {
                "height": height,
                "width": width,
            }
        )
        image = self.pipeline(
            prompt=inputs,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            negative_prompt=negative_prompt,
            **dims_dict,
        ).images[0]
        # Update the image export format - this is used when converting the image to proto, etc.
        image.format = image_format
        return CaptionedImage(
            output=caikit_dm.Image(image),
            caption=inputs,
        )

    @staticmethod
    def _force_to_nearest_multiples_of_eight(num_dict: Dict[str, int]):
        """Get the nearest multiple of eight for required params, e.g., height and width.

        Args:
            num_dict: dict[str, int]
                A dictionary whose values must be multiples of 8 for diffuser inference.
        returns:
            dict[str, int]
                A handle to the dictionary whose values are all multiples of 8.
        """
        for num_name, num_val in num_dict.items():
            # Force anything that is not a multiple of 8 to a multiple of 8, which is required
            if num_val // 8 != num_val / 8:
                log.warning(
                    f"Forcing inference param [{num_name}] to nearest multiple of 8"
                )
                num_dict[num_name] = round(num_val / 8) * 8
        return num_dict

    @classmethod
    def _get_device(cls, device: Optional[str]) -> Union[str, None]:
        """Get the device which we expect to run our models on. Defaults to GPU
        if one is available, otherwise falls back to None (cpu).

        NOTE: This code is adapted from Caikit NLP.

        Args:
            device: Optional[Union[str, int]]
                Device to be leveraged; if set to cls._DETECT_DEVICE, infers the device,
                otherwise we simply echo the value, which generally indicates a user override.

        Returns:
            Optional[str]
                Device string that we should move our models / tensors .to() at inference
                time.
        """
        if device == cls._DETECT_DEVICE:
            device = "cuda" if torch.cuda.is_available() else None
        return device

    def _validate_image_format(image_format: str):
        """Validates that the provided image format is an allowed choice.

        Args:
            image_format: str
                Image format to be used at serialization time for the data model.
        """
        # Initialize PIL's save driver registry if it isn't already
        if not SAVE:
            init()
        fmt = image_format.upper()
        if fmt not in SAVE:
            error(
                "<CCV14828291E>",
                KeyError(
                    f"Format {fmt} is unsupported! Supported formats: {list(SAVE.keys())}"
                ),
            )

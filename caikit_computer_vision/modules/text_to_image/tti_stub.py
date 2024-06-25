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
"""Stub module for text to image for testing runtime interfaces.
"""
# Standard
from typing import Union, get_args
import os

# Third Party
import numpy as np

# First Party
from caikit.core.modules import ModuleBase, ModuleConfig, ModuleSaver, module
from caikit.interfaces.vision import data_model as caikit_dm
import alog

# Local
from ...data_model import TextToImageResult
from ...data_model.tasks import TextToImageTask

log = alog.use_channel("TTI_STUB")


@module(
    id="28aa938b-1a33-11a0-11a3-bb9c3b1cbb11",
    name="Stub module for Text to Image",
    version="0.1.0",
    task=TextToImageTask,
)
class TTIStub(ModuleBase):
    def __init__(
        self,
        model_name,
    ) -> "TTIStub":
        log.debug("STUB - initializing text to image instance")
        super().__init__()
        self.model_name = model_name

    @classmethod
    def load(cls, model_path: Union[str, "ModuleConfig"]) -> "TTIStub":
        config = ModuleConfig.load(model_path)
        return cls.bootstrap(config.model_name)

    @classmethod
    def bootstrap(cls, model_name: str) -> "TTIStub":
        return cls(model_name)

    def save(self, model_path: str):
        saver = ModuleSaver(
            self,
            model_path=model_path,
        )
        with saver:
            saver.update_config({"model_name": self.model_name})

    def run(self, inputs: str, height: int, width: int) -> TextToImageResult:
        """Generates an image matching the provided height and width."""
        log.debug("STUB - running text to image inference")
        r_channel = np.full((height, width), 0, dtype=np.uint8)
        g_channel = np.full((height, width), 100, dtype=np.uint8)
        b_channel = np.full((height, width), 200, dtype=np.uint8)
        img = np.stack((r_channel, g_channel, b_channel), axis=2)
        return TextToImageResult(
            output=caikit_dm.Image(img),
        )

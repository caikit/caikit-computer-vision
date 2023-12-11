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
"""Initializes the local models directory and creates an example caikit 
computer vision object detector model (yolos-tiny transformer-based model)
for demo purposes.

This model can be loaded for inference, and it can also be used to demonstrate
the train interface for runtime.
"""
# Standard
from shutil import rmtree
import os

# Third Party
from common import DEMO_MODEL_ID, MODELS_DIR
from PIL import Image
import requests

# Local
from caikit_computer_vision.modules.object_detection import TransformersObjectDetector
import caikit_computer_vision


def init_models_dir():
    # Delete the models dir if it exists
    if os.path.isdir(MODELS_DIR):
        rmtree(MODELS_DIR)

    # Create the demo model in memory
    print("Bootstrapping tiny object detector model...")
    model = TransformersObjectDetector.bootstrap("hustvl/yolos-tiny")
    # Make sure we can run inference on the local model
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    pred = model.run(image)
    print("Bootstrapped model and ran example inference call for validation")

    print("Exporting tiny object detector model...")
    demo_model_path = os.path.join(MODELS_DIR, DEMO_MODEL_ID)
    model.save(demo_model_path)
    print(f"Exported model to [{demo_model_path}] successfully")


if __name__ == "__main__":
    init_models_dir()

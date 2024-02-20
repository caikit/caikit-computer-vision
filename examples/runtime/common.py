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
from pathlib import Path
import os
import sys

# Third Party
import yaml

# Add caikit computer vision to the syspath
PROJECT_ROOT = os.path.join(Path(__file__).parent.parent.parent)
sys.path.append(PROJECT_ROOT)

# Local
import caikit_computer_vision

# Constants that are generally useful most places
RUNTIME_CONFIG_PATH = os.path.join(PROJECT_ROOT, "runtime_config.yaml")
with open(RUNTIME_CONFIG_PATH, "r") as f:
    RUNTIME_CONFIG = yaml.safe_load(f)

TRAINING_DATA_DIR = "train_data"
TRAINING_IMG_DIR = os.path.join(TRAINING_DATA_DIR, "images")
TRAINING_LABELS_FILE = os.path.join(TRAINING_DATA_DIR, "labels.txt")

# Pull models dir out of runtime config for convenience
MODELS_DIR = RUNTIME_CONFIG["runtime"]["local_models_dir"]
# Directory dump_protos.py writes proto defs into
PROTO_EXPORT_DIR = "protos"

# Example model ID to use when creating a small model at model dir init time
DEMO_MODEL_ID = "my_model"
# New model that we are going to train and run an inference call on
NEW_MODEL_ID = "new_model"

RUNTIME_PORT = 8085

import os
from pathlib import Path
import sys
import yaml

# Add caikit computer vision to the syspath
PROJECT_ROOT = os.path.join(Path(__file__).parent.parent.parent)
sys.path.append(PROJECT_ROOT)
# Make sure we can import without issue
import os
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

RUNTIME_PORT = 8085
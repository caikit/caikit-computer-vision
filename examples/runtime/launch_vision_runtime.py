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
"""Launches the vision runtime.
"""
# Standard
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import tempfile

# Third Party
from create_models_dir import init_models_dir
from create_train_data import init_train_data
from dump_protos import export_protos

# First Party
from caikit.config.config import get_config
from caikit.runtime.__main__ import main
import alog
import caikit


def launch_runtime():
    alog.configure(default_level="debug")
    parser = argparse.ArgumentParser(description="Launch the vision runtime")
    parser.add_argument(
        "--purge_dirs",
        action="store_true",
        help="Deletes and reinitializes existing models and protos",
    )
    args = parser.parse_args()
    if args.purge_dirs:
        init_models_dir()
        init_train_data()
        export_protos()
    main()


if __name__ == "__main__":
    launch_runtime()

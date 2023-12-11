"""This demo is roughly derived from the example with the test fixture.
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

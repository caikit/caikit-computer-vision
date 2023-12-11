"""Writes the runtime protos out to a local directory; inspecting the output of this script
is helpful for understanding the API for the REST / gRPC servers.
"""

# Standard
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
from shutil import rmtree
import yaml

# First Party
import alog

# Local
from caikit.runtime.dump_services import dump_grpc_services, dump_http_services
import caikit
from common import PROTO_EXPORT_DIR, RUNTIME_CONFIG

def export_protos():
    # Make caikit_computer_vision available for import
    if os.path.isdir(PROTO_EXPORT_DIR):
        rmtree(PROTO_EXPORT_DIR)
    # Configure caikit runtime
    caikit.config.configure(config_dict=RUNTIME_CONFIG)
    # Dump proto files
    dump_grpc_services(output_dir=PROTO_EXPORT_DIR)
    dump_http_services(output_dir=PROTO_EXPORT_DIR)

if __name__ == "__main__":
    export_protos()

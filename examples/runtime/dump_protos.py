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
"""Writes the runtime protos out to a local directory; inspecting the output of this script
is helpful for understanding the API for the REST / gRPC servers.
"""

# Standard
from inspect import signature
from pathlib import Path
from shutil import rmtree
import json
import os
import shutil
import sys
import tempfile

# Third Party
from common import PROTO_EXPORT_DIR, RUNTIME_CONFIG
import yaml

# First Party
from caikit.runtime.dump_services import dump_grpc_services, dump_http_services
import alog
import caikit


def export_protos():
    # Make caikit_computer_vision available for import
    if os.path.isdir(PROTO_EXPORT_DIR):
        rmtree(PROTO_EXPORT_DIR)
    # Configure caikit runtime
    caikit.config.configure(config_dict=RUNTIME_CONFIG)
    # Export gRPC services first...
    # grpc service dumper kwargs depend on the version of caikit we are using
    grpc_service_dumper_kwargs = {
        "output_dir": PROTO_EXPORT_DIR,
        "write_modules_file": True,
    }
    # Only keep things in the signature, e.g., old versions don't take write_modules_file
    expected_grpc_params = signature(dump_grpc_services).parameters
    grpc_service_dumper_kwargs = {
        k: v for k, v in grpc_service_dumper_kwargs.items() if k in expected_grpc_params
    }
    dump_grpc_services(**grpc_service_dumper_kwargs)
    # Then export HTTP services...
    dump_http_services(output_dir=PROTO_EXPORT_DIR)


if __name__ == "__main__":
    export_protos()

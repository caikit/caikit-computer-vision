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
from os import path
import sys

# First Party
import alog

# Since the `caikit_template` package is not installed and it is not present in path,
# we are adding it directly
sys.path.append(
    path.abspath(path.join(path.dirname(__file__), "../../"))
)

# Local
import caikit_template
import caikit
from caikit.runtime import grpc_server

# Load configuration for model(s) serving
CONFIG_PATH = path.realpath(
    path.join(path.dirname(__file__), "config.yml")
)
caikit.configure(CONFIG_PATH)

alog.configure(default_level="debug")

# Load the model(s) in a gRPC server
grpc_server.main()

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

# Third Party
import grpc
from os import path
import sys

# Local
import caikit
from caikit.runtime.service_factory import ServicePackageFactory

# Since the `caikit_template` package is not installed and it is not present in path,
# we are adding it directly
sys.path.append(
    path.abspath(path.join(path.dirname(__file__), "../../"))
)

from caikit_template.data_model.hello_world import HelloWorldInput

# Load configuration for Caikit runtime
CONFIG_PATH = path.realpath(
    path.join(path.dirname(__file__), "config.yml")
)
caikit.configure(CONFIG_PATH)

training_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.TRAINING,
)

port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")
client_stub = training_service.stub_class(channel)

## Create request
training_data = path.realpath(path.join("..", "train_data", "sample_data.csv"))
print("train data:", training_data)
request = training_service.messages.HelloWorldTaskHelloWorldModuleTrainRequest(
    training_data={"file": {"filename": training_data}}, model_name="hello_world",
)

## Kick off training from server
response = client_stub.HelloWorldTaskHelloWorldModuleTrain(request)

print("RESPONSE:", response)
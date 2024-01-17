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
"""Runs a sample train [currently a stub], and exports the trained model under a new ID.
"""

# Standard
from pathlib import Path
from time import sleep
import os
import sys

# Third Party
from common import (
    DEMO_MODEL_ID,
    MODELS_DIR,
    TRAINING_DATA_DIR,
    TRAINING_IMG_DIR,
    TRAINING_LABELS_FILE,
)

# pylint: disable=no-name-in-module,import-error
try:
    # Third Party
    from generated import (
        computervisionservice_pb2_grpc,
        computervisiontrainingservice_pb2_grpc,
    )
except ImportError:
    raise ImportError("Failed to import cv service; did you compile your protos?")

# The location of these imported message types depends on the version of Caikit
# that we are using.
try:
    # Third Party
    from generated.caikit_data_model.caikit_computer_vision import (
        objectdetectiontrainset_pb2,
    )
    from generated.ccv import objectdetectiontaskrequest_pb2
    from generated.ccv import (
        objectdetectiontasktransformersobjectdetectortrainparameters_pb2 as odt_params_pb2,
    )
    from generated.ccv import (
        objectdetectiontasktransformersobjectdetectortrainrequest_pb2 as odt_request_pb2,
    )

    IS_LEGACY = False
except ModuleNotFoundError:
    # older versions of Caikit / py to proto create a flat proto structure
    # Third Party
    from generated import objectdetectiontaskrequest_pb2
    from generated import (
        objectdetectiontasktransformersobjectdetectortrainrequest_pb2 as odt_request_pb2,
    )
    from generated import objectdetectiontrainset_pb2

    IS_LEGACY = True

# Third Party
import grpc
import numpy as np

# First Party
from caikit.interfaces.vision import data_model as caikit_dm


### build the training request
# Training params; the only thing that changes between newer/older versions of caikit is that
# newer caikit versions pass all of these in under a parameters key and proto type, while old
# versions just pass them in directly.
def get_train_request():
    train_param_dict = {
        "model_path": os.path.join(MODELS_DIR, DEMO_MODEL_ID),
        "train_data": objectdetectiontrainset_pb2.ObjectDetectionTrainSet(
            img_dir_path=TRAINING_IMG_DIR,
            labels_file=TRAINING_LABELS_FILE,
        ),
        "num_epochs": 10,
        "learning_rate": 0.3,
    }
    if not IS_LEGACY:
        train_param_dict = {
            "parameters": odt_params_pb2.ObjectDetectionTaskTransformersObjectDetectorTrainParameters(
                **train_param_dict
            )
        }
    return odt_request_pb2.ObjectDetectionTaskTransformersObjectDetectorTrainRequest(
        model_name="new_model", **train_param_dict
    )


### Build the inference request
def get_inference_request():
    # For inference, just pick a random training image
    random_img_name = np.random.choice(os.listdir(TRAINING_IMG_DIR))
    with open(os.path.join(TRAINING_IMG_DIR, random_img_name), "rb") as f:
        im_bytes = f.read()

    return objectdetectiontaskrequest_pb2.ObjectDetectionTaskRequest(
        inputs=im_bytes,
        threshold=0,
    )


if __name__ == "__main__":

    # Setup the client
    port = 8085
    channel = grpc.insecure_channel(f"localhost:{port}")

    # send train request
    training_stub = (
        computervisiontrainingservice_pb2_grpc.ComputerVisionTrainingServiceStub(
            channel=channel
        )
    )
    response = training_stub.ObjectDetectionTaskTransformersObjectDetectorTrain(
        get_train_request()
    )
    print("*" * 30)
    print("RESPONSE from TRAIN gRPC\n")
    print(response)
    print("*" * 30)

    sleep(5)

    inference_stub = computervisionservice_pb2_grpc.ComputerVisionServiceStub(
        channel=channel
    )
    # NOTE: This just hits the old model, since normally the loading would be handled by something
    # like kserve/model mesh. But it might be more helpful to show how to manually load the model
    # and hit it here, just for reference.
    response = inference_stub.ObjectDetectionTaskPredict(
        get_inference_request(), metadata=[("mm-model-id", DEMO_MODEL_ID)], timeout=1
    )
    print("*" * 30)
    print("RESPONSE from INFERENCE gRPC\n")
    print(response)
    print("*" * 30)

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
"""Runs a sample train [currently a stub], and exports the trained model under a new ID,
then hits it with an inference request.
"""

# Standard
from time import sleep
import os

# Third Party
from common import (
    DEMO_MODEL_ID,
    MODELS_DIR,
    NEW_MODEL_ID,
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
        flatchannel_pb2,
        flatimage_pb2,
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
from PIL import Image
import grpc
import numpy as np


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
        model_name=NEW_MODEL_ID, **train_param_dict
    )


### Build the inference request
def get_inference_request():
    """Build an inference request. Here, not that `inputs` is a oneof; the corresponding
    proto class takes args named inputs_<type>. While we pass a flattened image as the arg
    here, we could equivalently pass the following args:

    Passing a path:
        ...
            inputs_str=os.path.join(TRAINING_IMG_DIR, random_img_name)

    Passing bytes:
        with open(os.path.join(TRAINING_IMG_DIR, random_img_name), "rb") as f:
            im_bytes = f.read()
        ...
            input_bytes = im_bytes
    """
    # For inference, just pick a random training image and read it as a BGR np arr
    random_img_name = np.random.choice(os.listdir(TRAINING_IMG_DIR))
    random_img = np.array(Image.open(os.path.join(TRAINING_IMG_DIR, random_img_name)))
    # Build the flattened channels; this is the same as from_numpy() on the flat image DM class
    flat_channels = [
        flatchannel_pb2.FlatChannel(values=random_img[:, :, ch_idx].flatten().tolist())
        for ch_idx in range(random_img.shape[-1])
    ]
    # And from the flattened channels, build the flat image
    flat_image = flatimage_pb2.FlatImage(
        flat_channels=flat_channels, image_shape=random_img.shape
    )

    return objectdetectiontaskrequest_pb2.ObjectDetectionTaskRequest(
        inputs_flatimage=flat_image,
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

    # The train command is basically an alias to save here - by default, if lazy_load_local_models
    # is True in the module, config, we sync new models from the model dir every
    # lazy_load_poll_period_seconds, which by deafult is 10 seconds. So 15 should be plenty of time
    # for the new_model to export and load.
    sleep(15)

    inference_stub = computervisionservice_pb2_grpc.ComputerVisionServiceStub(
        channel=channel
    )
    # NOTE: if this fails, make sure lazy_load_local_models is true in the config.
    # If needed, increase the log.level in the runtime config; setting level to
    # debug2 or higher should show polling of the local model dir, load calls, etc.
    response = inference_stub.ObjectDetectionTaskPredict(
        get_inference_request(), metadata=[("mm-model-id", NEW_MODEL_ID)], timeout=1
    )
    print("*" * 30)
    print("RESPONSE from INFERENCE gRPC\n")
    print(response)
    print("*" * 30)

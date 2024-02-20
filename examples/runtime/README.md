# Caikit Computer Vision Runtime Example

This example shows how to leverage the gRPC interface a runtime instance of caikit computer vision.

## Quick Start:

1. Start the vision runtime with the purge directory flag:

```bash
python3 launch_vision_runtime.py --purge_dirs
```

This will blow away any existing local models / protos, and recreate the defaults, i.e.,

- `models`: contains a single tiny YOLOS model with the ID `my_model`
- `protos`: contains the dumped proto files for the data model; note that this includes the train service
- `train_data`: randomly generated jpegs with a label format; this format will likely be changed in the future, and is currently only used for demo purposes

2. In a separate terminal window run the grpcio-tools are installed into your virtual environment and compile your proto files with `protoc`, e.g., as shown below:

```bash
pip3 install grpcio-tools

python3 -m grpc_tools.protoc -I protos --python_out=generated --grpc_python_out=generated protos/*.proto
```

Inspect the `generated` python package; you should see lots of `pb2.py/pb2_grpc.py` files!

3. Call the script to train a new model and run inference against it, i.e., by running:

```bash
python3 run_train_and_inference.py
```

This will create a gRPC client to call `.train()` on the object detector module, which currently a stub. Note that the training parameters are not actually being used here and only are present for illustration purposes to show how args are passed. The result of the train job is simply the same model provided as the base, exported under a new model ID, `new_model`.

Then, an inference request is sent to the newly loaded model, which calls `.run()`, to produce the prediction that is logged by the script.

NOTE: in order to hit the new model, we need to set `runtime.lazy_load_local_models=True` in the runtime config, which by default will sync the local model dir with the in memory runtime (i.e., load models that have been added and unload models that have been deleted) every 10 seconds. If inference fails, try setting the log level to `debug2` and ensure that you see the runtime polling for new models periodically.

## A Deeper Look

Now that we've run a successful call to a `.train()` and `.run()` through runtime, it's useful to look at how these scripts work internally. More specifically:

1. What do the proto message definitions generated from our data model objects look like?
2. What do the proto message definitions generated off of `.run()` and `.train()` look like?
3. What do the gRPC services that are generated off of `.run()` and `.train()` look like?

### Protos For Data Model Classes

Let's start with the proto definitions for our data model objects. For the sake of simplicity, we'll just consider one type of object,`ObjectDetectionResult`; the type produced by our object detection module.

```python
@dataobject(package="caikit_data_model.caikit_computer_vision")
class ObjectDetectionResult(DataObjectBase):
    detected_objects: Annotated[List[DetectedObject], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]
```

To export the `.proto` files, run `python3 dump_protos.py`. This will delete your existing `protos` directory and recreate it. Inspecting the contents of the `protos` folder; you should see the following:

- A file named `openapi.json`
- Lots of `.proto` files, including one of the following:
  - More recent versions of Caikit: `caikit_data_model.caikit_computer_vision.objectdetectionresult.proto`
  - Older versions of Caikit: `objectdetectionresult.proto`

Regardless of its name, this file contains the `ObjectDetectionResult` data model class, shown below:

```protobuf
...
message ObjectDetectionResult {
  repeated caikit_data_model.caikit_computer_vision.DetectedObject detected_objects = 1;
  caikit_data_model.common.ProducerId producer_id = 2;
}
```

### Proto Messages Types

#### Inference & Task Messages

Now that we understand how our defined data model object can be translated and/or exported to a `.proto` message, it's helpful to understand how task, training, and serving message/service definitions are formed, as well as the naming conventions that they follow. Starting with the task definition, which can be found in under the `data_model/tasks.py` of `caikit_computer_vision.py`:

```python
@task(
    required_parameters={"inputs": bytes},
    output_type=ObjectDetectionResult,
)
class ObjectDetectionTask(TaskBase):
    """The Object Detection Task is responsible for taking an input image
    and producing 0 or more detected objects, which typically include labels
    and confidence scores.
    """
```

This task definition is currently being implemented by the transformers based object detector module. **This is important, because if you don't have a module for a task, you won't see the proto definition for it exported.** The reason is that the library inspects the type annotations of `.run()` on the implementing module, which must be compatible with the parameters we outline in our task.

The declaration of `.run()` for the transformer-based module is shown below:

```python
...
    def run(
        self, inputs: image_pil_backend.PIL_SOURCE_TYPES, threshold: float = 0.5
    ) -> ObjectDetectionResult:
```

Where `image_pil_backend.PIL_SOURCE_TYPES` is a union of types that can be resolved into a PIL image, one of which is `bytes`. As such, the `.run()` declaration is compatible with the task declaration; because of the way the task is defined, the runtime expects `inputs` to be of type `bytes`, with `threshold` as an optional float parameter. We can find the message type defining exactly this in the task request definition proto file, which is one of the following:

- More recent versions of Caikit: `ccv.objectdetectiontaskrequest.proto`; here, ccv is the `service_generation` name defined in our runtime config
- Older versions of Caikit: `objectdetectiontaskrequest.proto`

```protobuf
...
message ObjectDetectionTaskRequest {
  bytes inputs = 1;
  optional double threshold = 2;
}
```

Note: `float` in Python maps to `double` in proto, as defined [here](https://protobuf.dev/programming-guides/proto3/#scalar).

#### What About Conflicts?

This brings up a question - if parameters defined outside of the task definition are inferred by runtime as optional, what happens if there are conflicts?

The short answer is that currently, you'll see a warning about the conflict, and that proto file won't be generated. It's easy to check this behavior by creating a new module.

```python
@module(
    id="28dc918b-4e19-41c3-22a1-aa9c3c5caa17",
    name="Foo Bar",
    version="0.1.0",
    task=ObjectDetectionTask,
)
class TransformersObjectDetector(ModuleBase):
    def run(
        self, inputs: image_pil_backend.PIL_SOURCE_TYPES, threshold: str = "foo"
    ) -> ObjectDetectionResult:
        raise NotImplementedError("Not a real module")
```

Notice that here, `threshold` was changed to a `str` which causes a type collision! If you run the proto dump script, you'll see the following warning message in the log:

```
Cannot generate task rpc for <class 'caikit_computer_vision.data_model.tasks.ObjectDetectionTask'>: Conflicting value types for arg threshold: <class 'str'> != <class 'float'>
```

and you won't have a `ccv.objectdetectiontaskrequest.proto`/`objectdetectiontaskrequest.proto` in the dumped protos.

#### Train Messages

The story for training is similar; the definitions for proto messages are derived from type hints on the implementation for `.train` on the module of interest.

Currently, the stub example for `.train` on the object detector class is defined as follows:

```python
    @classmethod
    def train(
        cls,
        model_path: str,
        train_data: ObjectDetectionTrainSet,
        num_epochs: int,
        learning_rate: float
    ):
```

Where `ObjectDetectionTrainSet` is a data model object. The train message file name is built with the following name, in all lowercase letters:
`{{service_gen_name}}.{{task_name}}task{{impl_class_name}}trainrequest.proto`
where:

- `{{service_gen_name}}` is the name of your service generation key from your runtime config, e.g., `ccv` As before, this is only used for more recent versions of Caikit; in older versions, the leading `{{service_gen_name}}.` is omitted.
- `{{task_name}}` is the name of the task, in this case `objectdetection`
- `{{impl_class_name}}` is the name of the implementing module class being considered, in this cases `transformersobjectdetector`

As a result we get a file named one of the following:

- More recent versions of Caikit: `ccv.objectdetectiontasktransformersobjectdetectortrainrequest.proto`
- Older versions of Caikit: `objectdetectiontasktransformersobjectdetectortrainrequest.proto`

which contains the message definition to trigger a train request.

```protobuf
message ObjectDetectionTaskTransformersObjectDetectorTrainRequest {
  string model_name = 1;
  caikit_data_model.common.S3Path output_path = 2;
  string model_path = 3;
  caikit_data_model.caikit_computer_vision.ObjectDetectionTrainSet train_data = 4;
  int64 num_epochs = 5;
  double learning_rate = 6;
}
```

Notice that the first two fields are special; `model_name` is the model ID that will be used for this job in runtime, and `output_path` specifies some s3 location to export the model to, which may be as simple as a mounted path on disk. the remaining fields are our arguments to `.train` method.

### Service Definitions

Now that we understand what messages are created by which parts of the Python data model / module code and the naming conventions that they follow, we can talk about the gRPC services. There are two; one for inference, and one for training.

The inference service is contained in: `computervisionservice.proto`

```protobuf
...
service ComputerVisionService {
  rpc ObjectDetectionTaskPredict(caikit.runtime.ComputerVision.ObjectDetectionTaskRequest) returns (caikit_data_model.caikit_computer_vision.ObjectDetectionResult);
}
```

And the train service is contained in: `computervisiontrainingservice.proto`

```protobuf
...
service ComputerVisionTrainingService {
  rpc ObjectDetectionTaskTransformersObjectDetectorTrain(caikit.runtime.ComputerVision.ObjectDetectionTaskTransformersObjectDetectorTrainRequest) returns (caikit_data_model.runtime.TrainingJob);
}
```

The contents of these services should now be unsurprising; the inference service defines RPCs where the arg / return type encapsulates a request whose types are derived from `.run` and the task definition as we saw before. The argument for `.train` is similar, and the return type indicates information about the training job that we created.

For an example of calling both of these services with a gRPC client, see `run_train_and_inference.py`.

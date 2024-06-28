# Text To Image (SDXL)
This directory provides guidance for running text to image inference for text to image and a few useful scripts for getting started.

## Task and Module Overview
The text to image task has only one required parameter, the input text, and produces a `caikit_computer_vision.data_model.CaptionedImage` in response, which wraps the provided input text, as well as the generated image.

Currently there are two modules for text to image.:
- `caikit_computer_vision.modules.text_to_image.TTIStub` - A simple stub which produces a blue image of the request height and width at inference. This module is purely used for testing purposes.

- `caikit_computer_vision.modules.text_to_image.SDXL` - A module implementing text to image via SDXL.

This document will help you get started with both at the library & runtime level, ending with a sample gRPC client that can be usde to hit models running in a Caikit runtime container.

## Building the Environment
The easiest way to get started is to build a virtual environment in the root directory of this repo. Make sure the root of this project is on the `PYTHONPATH` so that `caikit_computer_vision` is findable.

To install the project:
```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

Note that if you prefer running in Docker, you can build an image as you normally would, and mount things into a running container:
```bash
docker build -t caikit-computer-vision:latest .
```

## Creating the Models
For the remainder of this demo, commands are intended to be run from this directory. First, we will be creating our models & runtime config in a directory named `caikit`, which is convenient for running locally or mounting into a container.

Copy the runtime config from the root of this project into the `caikit` directory.
```bash
mkdir -p caikit/models
cp ../../runtime_config.yaml caikit/runtime_config.yaml
```

Next, create your models.
```bash
python create_tti_models.py
```

This will create two models.
1. The stub model, at `caikit/models/stub_model`
2. The SDXL turbo model, at `caikit/models/sdxl_turbo_model`

Note that the names of these directories will be their model IDs in caikit runtime.

## Running Local Inference / API Overview
The text to image API is simple.

### Stub Module
For the stub module, we take an input prompt, a height, and a width, and create a blue image of the specified height and width.
```python
run(
    inputs: str, 
    height: int, 
    width: int
) -> CaptionedImage:
```

Example using the stub model created from above:
```python
>>> import caikit_computer_vision, caikit
>>> stub_model = caikit.load("caikit/models/stub_model")
>>> res = stub_model.run("This is a text", height=512, width=512)
```

The resulting object holds the provided input text under `.caption`:
```python
>>> res.caption
'This is a text'
```
And the image bytes stored as PNG under `.output.image_data`
```python
>>> res.output.image_data
b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x02\x00\x00\x00\x02\x00 ...
```
Note that the `output` object is a `Caikit` image backed by PIL. If you need a handle to it, you can call `as_pil()` to get handle to the PIL object as shown below.
```
>>> pil_im = res.output.as_pil()
>>> type(pil_im)
<class 'PIL.Image.Image'>
```

Grabbing a handle to the PIL image and then `.save()` on the result is the easiest way to save the image to disk.

### SDXL Module
The SDXL module is signature to the stub, with some additional options.
```python
run(
    inputs: str,
    height: int = 512,
    width: int = 512,
    num_steps: int = 1,
    guidance_scale: float = 0.0,
    negative_prompt: Optional[str] = None,
    image_format: str = "png",
) -> CaptionedImage:
```

A full description of these args can be seen with `help(caikit_computer_vision.modules.text_to_image.SDXL.run)`. Notably, for SDXL turbo, guidance scale and negative prompt should be left as the defaults, as they were not used to train this model.

The `image_format` arg follows the same conventions as PIL and controls the format of the serialized bytes. An example for this module similar to the previous one is shown below, where we generate a picture of a puppy in a field, storing the image in jpeg format for serialization purposes.

```python
>>> import caikit_computer_vision, caikit
>>> stub_model = caikit.load("caikit/models/sdxl_turbo_model")
>>> res = stub_model.run("A golden retriever puppy sitting in a grassy field", height=512, width=512, num_steps=2, image_format="jpeg")
```


## Inference Through Runtime
To write a client, you'll need to export the proto files to compile. To do so, run `python export_protos.py`; this will use the runtime file you had previously copied to create a new directory called `protos`, containing the exported data model / task protos from caikit runtime.

Then to compile them, you can do something like the following; note that you may need to `pip install grpcio-tools` if it's not present in your environment, since it's not a dependency of `caikit_computer_vision`:
```bash
python -m grpc_tools.protoc -I protos --python_out=generated --grpc_python_out=generated protos/*.proto
```

In general, you will want to run Caikit Runtime in a Docker container. The easiest way to do this is to mount the `caikit` directory with your models into the container as shown below.
```bash
docker run -e CONFIG_FILES=/caikit/runtime_config.yaml \
    -v $PWD/caikit/:/caikit \
    -p 8080:8080 -p 8085:8085  \
    caikit-computer-vision:latest python -m caikit.runtime
```

Then, you can hit it with a gRPC client using your compiled protobufs. A full example of inference via gRPC client calling both models can be found in `sample_client.py`.

Running `python sample_client.py` should produce two images.
- `stub_response_image.png` - blue image generated from the stub module
- `turbo_response_image.png` - picture of a golden retriever in a field generated by SDXL turbo

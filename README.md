# Caikit Template

GitHub Template with a boilerplate repository which serves an example AI model using [caikit](https://github.com/caikit/caikit).

## Before Starting

The following tools are required:

- [python](https://www.python.org) (v3.8+)
- [pip](https://pypi.org/project/pip/) (v23.0+)

**Note:** Before installing dependencies and to avoid conflicts in your environment, it is advisable to use a virtual environment. The subsection which follows provides an example of a virtual environment, python venv.

Install the dependencies: `pip install -r requirements.txt`

### (Optional) Setting Up Virtual Environment using Python venv

For [(venv)](https://docs.python.org/3/library/venv.html)], make sure you are in an activated `venv` when running `python` in the example commands that follow. Use `deactivate` if you want to exit the `venv`.

For example, to create and activate a virtual environment using `venv`:

```shell
python3 -m venv venv
source venv/bin/activate
```

## Repository Layout

```text
├── caikit-template/:                       top-level package directory (will change to your repo name after template is deployed)
│   │── caikit_template/:                   a directory that defines Caikit module(s) that can include algorithm(s) implementation that can train/run an AI model 
│   │   ├── config/:                        a directory that contains the configuration for the module and model input and output
│   │   │   ├── config.yml:                 configuration for the module and model input and output
│   │   ├── data_model/:                    a directory that contains the data format of the Caikit module
│   │   │   ├── hello_world.py:             data class that represents the AI model attributes in code
│   │   │   ├── __init__.py:                makes the hello_world class visible in the project
│   │   ├── modules/:                       a directory that contains the Caikit module of the model
│   │   │   ├── hello_world.py:             a class that bootstraps the AI model in Caikit so it can be served and used (infer/train)
│   │   │   ├── __init__.py:                makes the hello_world class visible in the project
|   |   |── __init__.py:                    makes the data_model and runtime_model packages visible
│   │── demo/:                              a directory which contains code and configuration to test the model
│   │   │── client/:                        a directory which contains artifacts to use (infer and train) the AI model spceified in the `caikit_template` package
|   │   │   ├── config.yml:                 caikit runtime configuration file
│   │   │   ├── infer_model.py:             sample client which calls the Caikit runtime to perform inference on a model it is serving
│   │   │   ├── train_model.py:             sample client which calls the Caikit runtime to perform training on a model it is serving
│   │   │── models/:                        a directory that contains the Caikit metadata of the models and any artifacts required to run the models (usually generated after saving and should not be modified)
│   │   │   ├── hello_world/config.yml:     a metadata that defines the example Caikit model
│   │   │── server/:                        a directory which contains artifacts to start Caikit runtime
|   │   │   ├── config.yml:                 configuration for handling the model by the Caikit runtime
│   │   │   ├── start_runtime.py:           a wrapper to start the Caikit runtime as a gRPC server. The runtime will load the model at startup
|   │   ├── train_data/:                    a directory which contains the training data
|   │   |   ├── sample_data.csv:            sample training dataset to perform training of the model
└── └── requirements.txt:                   specifies library dependencies
```

## Starting the Caikit Runtime

In one terminal, start the runtime server:

```shell
cd client
python3 start_runtime.py
```

You should see output similar to the following:

```ShellSession
$ python3 start_runtime.py

[...]
{"channel": "MODEL-LOADER", "exception": null, "level": "info", "log_code": "<RUN89713784I>", "message": "Singleton cache: '{}'", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.833744"}
{"channel": "MODEL-SIZER", "exception": null, "level": "info", "log_code": "<RUN62161564I>", "message": "No configured model size multiplier found for model type 'standalone-model' for model 'hello_world'. Using default multiplier '10.000000'", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.834439"}
{"channel": "GP-SERVICR-I", "exception": null, "level": "info", "log_code": "<RUN76773776I>", "message": "Metering is disabled, to enable set `metering.enabled` in config to true", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.834766"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit_template", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.834905"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.common", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.834977"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.runtime", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.835026"}
{"channel": "GP-SERVICR-I", "exception": null, "level": "info", "log_code": "<RUN76773778I>", "message": "Validated Caikit Library CDM successfully", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.835119"}
{"channel": "GP-SERVICR-I", "exception": null, "level": "info", "log_code": "<RUN76884779I>", "message": "Constructed inference service for library: caikit_template, version: unknown", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.836444"}
{"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN81194024I>", "message": "Intercepting RPC method /caikit.runtime.Template.TemplateService/HelloWorldTaskPredict", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.836523"}
{"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN33333123I>", "message": "Wrapping safe rpc for Predict", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.836663"}
{"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN30032825I>", "message": "Re-routing RPC /caikit.runtime.Template.TemplateService/HelloWorldTaskPredict from <function _ServiceBuilder._GenerateNonImplementedMethod.<locals>.<lambda> at 0x7fa470fdb550> to <function CaikitRuntimeServerWrapper.safe_rpc_wrapper.<locals>.safe_rpc_call at 0x7fa470ff55e0>", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.836722"}
[...]
{"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN24924908I>", "message": "Interception of service caikit.runtime.Template.TemplateService complete", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.836783"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit_template", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.836977"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.common", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.837052"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.runtime", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.837114"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit_template", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.837267"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.common", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.837320"}
{"channel": "COM-LIB-INIT", "exception": null, "level": "info", "log_code": "<RUN11997772I>", "message": "Loading service module: caikit.interfaces.runtime", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.837373"}
{"channel": "GT-SERVICR-I", "exception": null, "level": "info", "log_code": "<RUN76773777I>", "message": "Validated Caikit Library CDM successfully", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.837458"}
{"channel": "GT-SERVICR-I", "exception": null, "level": "info", "log_code": "<RUN76884779I>", "message": "Constructed train service for library: caikit_template, version: unknown", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.838420"}
{"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN81194024I>", "message": "Intercepting RPC method /caikit.runtime.Template.TemplateTrainingService/HelloWorldTaskHelloWorldModuleTrain", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.838478"}
{"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN33333123I>", "message": "Wrapping safe rpc for Train", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.838572"}
{"channel": "SERVER-WRAPR", "exception": null, "level": "info", "log_code": "<RUN30032825I>", "message": "Re-routing RPC /caikit.runtime.Template.TemplateTrainingService/HelloWorldTaskHelloWorldModuleTrain from <function _ServiceBuilder._GenerateNonImplementedMethod.<locals>.<lambda> at 0x7fa470ff53a0> to <function CaikitRuntimeServerWrapper.safe_rpc_wrapper.<locals>.safe_rpc_call at 0x7fa4602adaf0>", "num_indent": 0, "thread_id": 8604329472, "timestamp": "2023-06-09T11:22:12.838619"}
```

## Inferencing the Served Model

In another terminal, run the client code to infer the model:

```shell
cd client
python3 infer_model.py
```

The client code calls the model and queries for generated text using text passed from the client.

You should see output similar to the following after the word `World` is passed:

```ShellSession
$ python3 infer_model.py

RESPONSE: greeting: "Hello World"
```

## Training the Served Model

In another terminal, run the client code to train the model:

```shell
cd client
python3 train_model.py
```

The client code trains the model with sample data in `train_data/` and outputs the
trained model to `training_output/` by default.

You should see output similar to the following:

```ShellSession
$ python3 train_model.py

RESPONSE: training_id: "ace2fd4c-0a50-49ef-b4db-9d9bbe2eefaf"
model_name: "hello_world"
```

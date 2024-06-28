# Standard
from inspect import signature
from shutil import rmtree
import os

# First Party
from caikit.runtime.dump_services import dump_grpc_services
import caikit

SCRIPT_DIR=os.path.dirname(__file__)
PROTO_EXPORT_DIR=os.path.join(SCRIPT_DIR, "protos")
RUNTIME_CONFIG_PATH=os.path.join(SCRIPT_DIR, "caikit", "runtime_config.yaml")

if os.path.isdir(PROTO_EXPORT_DIR):
    rmtree(PROTO_EXPORT_DIR)
# Configure caikit runtime
caikit.config.configure(config_yml_path=RUNTIME_CONFIG_PATH)

# Export gRPC services...
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
# NOTE: If you need an http client for inference, use `dump_http_services` from caikit instead.
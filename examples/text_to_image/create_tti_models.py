"""Creates and exports SDXL Turbo as a caikit module.
"""
# Standard
import os

# Local
from caikit_computer_vision.modules.text_to_image import TTIStub

SCRIPT_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPT_DIR, "caikit", "models")
STUB_MODEL_PATH = os.path.join(MODELS_DIR, "stub_model")
SDXL_TURBO_MODEL_PATH = os.path.join(MODELS_DIR, "sdxl_turbo_model")

if not os.path.isdir(MODELS_DIR):
    os.mkdir(MODELS_DIR)

model = TTIStub.bootstrap("foobar")
model.save(STUB_MODEL_PATH)


# Third Party
### Make the model for SDXL turbo
import diffusers

# Local
from caikit_computer_vision.modules.text_to_image import SDXL

### Download the model for SDXL turbo...
sdxl_model = SDXL.bootstrap("stabilityai/sdxl-turbo")
sdxl_model.save(SDXL_TURBO_MODEL_PATH)
# Standard
# There appears to be a bug in the way that sharded safetensors are reloaded into the
# pipeline from diffusers, and there ALSO appears to be a bug where passing the max
# safetensor shard size to diffusers on a pipeline doesn't work as exoected.
#
# it is unfortunate that we need this workaround, but delete
# the sharded u-net, and reexport it as one file. By default the
# max shard size if 10GB, and the turbo unit is barely larger than 10.
from shutil import rmtree

unet_path = os.path.join(SDXL_TURBO_MODEL_PATH, "sdxl_model", "unet")
try:
    diffusers.UNet2DConditionModel.from_pretrained(unet_path)
except RuntimeError:
    print(
        "Unable to reload turbo u-net due to sharding issues; reexporting as single file"
    )
    rmtree(unet_path)
    sdxl_model.pipeline.unet.save_pretrained(unet_path, max_shard_size="12GB")

# Make sure the model can be loaded and that we can get an image out of it
reloaded_model = SDXL.load(SDXL_TURBO_MODEL_PATH)
cap_im = reloaded_model.run("A golden retriever sitting in a grassy field")
print("[DONE]")

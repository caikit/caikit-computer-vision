import io

from generated import (
    computervisionservice_pb2_grpc,
)
from generated.ccv import texttoimagetaskrequest_pb2

from PIL import Image
import grpc


# Setup the client
port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")

inference_stub = computervisionservice_pb2_grpc.ComputerVisionServiceStub(
    channel=channel
)

inference_request = texttoimagetaskrequest_pb2.TextToImageTaskRequest(
    inputs="A golden retriever sitting in a grassy field",
    height=512,
    width=512,
)

# Call to stub model...
response = inference_stub.TextToImageTaskPredict(
    inference_request, metadata=[("mm-model-id", "stub_model")], timeout=60
)
Image.open(io.BytesIO(response.output.image_data)).save("stub_response_image.png")

# Call to SDXL turbo model...
response = inference_stub.TextToImageTaskPredict(
    inference_request, metadata=[("mm-model-id", "sdxl_turbo_model")], timeout=60
)
Image.open(io.BytesIO(response.output.image_data)).save("turbo_response_image.png")

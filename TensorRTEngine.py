#pycuda not supported in my environment, so its the basic template
import requests
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from fastapi import FastAPI
from ray import serve

# Load the TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open('model.trt', 'rb') as f:
    engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)

# Define a function to run inference on the TensorRT engine
def infer(x: np.ndarray) -> np.ndarray:
    # Allocate device memory and create a CUDA context
    d_input = cuda.mem_alloc(x.nbytes)
    d_output = cuda.mem_alloc(1 * np.float32().nbytes)
    cuda_ctx = cuda.Device(0).make_context()
    
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, x, stream)

    # Create an execution context and set input/output bindings
    context = engine.create_execution_context()
    bindings = [int(d_input), int(d_output)]
    context.set_binding_shape(0, x.shape)
    context.set_binding_shape(1, (1,))
    
    # Run inference and copy output data from device memory
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(x, d_output, stream)
    
    # Clean up
    stream.synchronize()
    cuda_ctx.pop()

    return x

# Define a FastAPI app and wrap it in a deployment with a route handler
app = FastAPI()

@serve.deployment(route_prefix="/predict")
@serve.ingress(app)
class TensorRTDeployment:
    # FastAPI will automatically parse the HTTP request for us
    @app.post("/predict")
    async def predict(self, data: np.ndarray) -> np.ndarray:
        # Run inference on the TensorRT engine
        output = infer(data)
        return output

# Deploy the deployment
serve.run(TensorRTDeployment.bind())

# Query the deployment and print the result
data = np.random.randn(1, 3, 256, 256).astype(np.float32)
response = requests.post("http://localhost:8000/predict", json=data.tolist())
output = np.array(response.json())
print(output)

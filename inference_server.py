"""
Local inference server that speaks Triton's HTTP API (KServe v2 protocol).

Stands in for NVIDIA Triton on a Mac: same URL, same JSON in and out, so the
twin code would work unchanged against a real Triton server later.
Runs the ONNX model with onnxruntime (native on Apple Silicon).

    python inference_server.py          -> http://localhost:8000
"""
import numpy as np
import onnxruntime as ort
from aiohttp import web

MODEL_NAME = "temperature_model"
MODEL_PATH = f"models/{MODEL_NAME}/1/model.onnx"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
IN_NAME = session.get_inputs()[0].name
OUT_NAME = session.get_outputs()[0].name


async def ready(request):
    return web.json_response({"ready": True})


async def model_ready(request):
    if request.match_info["model"] != MODEL_NAME:
        raise web.HTTPNotFound(text="unknown model")
    return web.json_response({"name": MODEL_NAME, "ready": True})


async def infer(request):
    if request.match_info["model"] != MODEL_NAME:
        raise web.HTTPNotFound(text="unknown model")
    body = await request.json()
    inp = body["inputs"][0]
    x = np.asarray(inp["data"], dtype=np.float32).reshape(inp["shape"])
    (y,) = session.run([OUT_NAME], {IN_NAME: x})
    return web.json_response({
        "model_name": MODEL_NAME,
        "outputs": [{
            "name": OUT_NAME,
            "datatype": "FP32",
            "shape": list(y.shape),
            "data": y.flatten().tolist(),
        }],
    })


app = web.Application()
app.add_routes([
    web.get("/v2/health/ready", ready),
    web.get("/v2/models/{model}/ready", model_ready),
    web.post("/v2/models/{model}/infer", infer),
])

if __name__ == "__main__":
    print(f"serving {MODEL_NAME} on http://localhost:8000")
    web.run_app(app, host="localhost", port=8000, print=None)

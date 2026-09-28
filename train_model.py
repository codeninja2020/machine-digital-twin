"""
Train a tiny machine-health model and export it as ONNX for Triton.

The tutorial never supplies a model (and an object-detection model would not
accept [temperature, vibration] anyway), so this builds one:
logistic regression on synthetic data, fitted with numpy, written as an
ONNX graph: output_0 = sigmoid(input_0 @ W + b)

Needs only: pip install numpy onnx
Writes:      models/temperature_model/1/model.onnx
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

rng = np.random.default_rng(42)

# --- 1. Synthetic sensor data ------------------------------------------------
n = 5000
temp = rng.uniform(60, 95, n)
vib = rng.uniform(0.1, 0.6, n)
# "Ground truth": a machine is unhealthy when it runs hot AND shakes
risk = 0.25 * (temp - 82) + 18 * (vib - 0.42)
label = (risk + rng.normal(0, 0.8, n) > 0).astype(np.float64)
X = np.column_stack([temp, vib])

# --- 2. Fit logistic regression on standardised features --------------------
mu, sd = X.mean(axis=0), X.std(axis=0)
Xs = (X - mu) / sd
w, b = np.zeros(2), 0.0
for _ in range(3000):
    p = 1 / (1 + np.exp(-(Xs @ w + b)))
    w -= 0.5 * Xs.T @ (p - label) / n
    b -= 0.5 * np.mean(p - label)
acc = np.mean((p > 0.5) == label)
print(f"training accuracy: {acc:.3f}")

# Fold the standardisation into the weights so the model takes raw readings
W_raw = (w / sd).reshape(2, 1).astype(np.float32)
b_raw = np.array([b - np.sum(w * mu / sd)], dtype=np.float32)

# --- 3. Build the ONNX graph -------------------------------------------------
inp = helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch", 2])
out = helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch", 1])
graph = helper.make_graph(
    nodes=[
        helper.make_node("MatMul", ["input_0", "W"], ["z0"]),
        helper.make_node("Add", ["z0", "b"], ["z"]),
        helper.make_node("Sigmoid", ["z"], ["output_0"]),
    ],
    name="machine_health",
    inputs=[inp],
    outputs=[out],
    initializer=[numpy_helper.from_array(W_raw, "W"),
                 numpy_helper.from_array(b_raw, "b")],
)
# opset 13 / IR 8 keeps it loadable by older Triton releases too
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)

path = os.path.join("models", "temperature_model", "1", "model.onnx")
os.makedirs(os.path.dirname(path), exist_ok=True)
onnx.save(model, path)
print(f"saved {path}")

# --- 4. Sanity check ---------------------------------------------------------
for t, v in [(65, 0.15), (80, 0.35), (90, 0.5)]:
    z = t * W_raw[0, 0] + v * W_raw[1, 0] + b_raw[0]
    print(f"temp={t} vib={v} -> fault probability {1 / (1 + np.exp(-z)):.2f}")

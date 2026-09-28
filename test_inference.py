"""Check the inference server answers before running the twin."""
import requests

URL = "http://localhost:8000/v2/models/temperature_model/infer"
for t, v in [(65, 0.15), (80, 0.35), (90, 0.5)]:
    payload = {"inputs": [{"name": "input_0", "shape": [1, 2],
                           "datatype": "FP32", "data": [t, v]}]}
    r = requests.post(URL, json=payload, timeout=5)
    r.raise_for_status()
    p = r.json()["outputs"][0]["data"][0]
    print(f"temp={t} vib={v} -> p(fault)={p:.2f}")

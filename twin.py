"""
The digital twin: an OpenUSD stage kept in sync with live sensor data + AI.

  sensor (ws :8765) --> twin.py --> inference server (http :8000)
                           |
                           +--> twin.usda  (the USD stage, saved on every update)
                           +--> viewer     (http://localhost:8080, live in your browser)

The USD stage is the source of truth: the viewer is sent values READ BACK
from the stage, exactly as an Omniverse viewport renders what is on the stage.

    python twin.py
"""
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from aiohttp import web
from pxr import Gf, Sdf, Usd, UsdGeom

SENSOR_URL = "ws://localhost:8765"
INFER_URL = "http://localhost:8000/v2/models/temperature_model/infer"
PRIM_PATH = "/World/Machine_1"
STAGE_FILE = Path("twin.usda")
THRESHOLD = 0.8

HEALTHY = Gf.Vec3f(0.10, 0.75, 0.30)
FAULT = Gf.Vec3f(0.90, 0.10, 0.05)

viewers: set = set()


# ---------------------------------------------------------------- USD stage --
def build_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(STAGE_FILE)) if not STAGE_FILE.exists() \
        else Usd.Stage.Open(str(STAGE_FILE))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube = UsdGeom.Cube.Define(stage, PRIM_PATH)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([HEALTHY])

    prim = cube.GetPrim()
    for name, typ in [("twin:temperature", Sdf.ValueTypeNames.Float),
                      ("twin:vibration", Sdf.ValueTypeNames.Float),
                      ("twin:faultProbability", Sdf.ValueTypeNames.Float),
                      ("twin:status", Sdf.ValueTypeNames.String),
                      ("twin:lastUpdate", Sdf.ValueTypeNames.Double)]:
        prim.CreateAttribute(name, typ, custom=True)
    stage.Save()
    return stage


def update_stage(stage, reading: dict, prob: float) -> None:
    prim = stage.GetPrimAtPath(PRIM_PATH)
    status = "FAULT" if prob > THRESHOLD else "HEALTHY"
    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(
        [FAULT if status == "FAULT" else HEALTHY])
    prim.GetAttribute("twin:temperature").Set(reading["temperature"])
    prim.GetAttribute("twin:vibration").Set(reading["vibration"])
    prim.GetAttribute("twin:faultProbability").Set(prob)
    prim.GetAttribute("twin:status").Set(status)
    prim.GetAttribute("twin:lastUpdate").Set(time.time())
    stage.Save()


def read_stage(stage) -> dict:
    """What a renderer would see: values read back from the stage."""
    prim = stage.GetPrimAtPath(PRIM_PATH)
    color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()[0]
    return {
        "prim": PRIM_PATH,
        "color": [round(c, 3) for c in color],
        "temperature": prim.GetAttribute("twin:temperature").Get(),
        "vibration": prim.GetAttribute("twin:vibration").Get(),
        "faultProbability": prim.GetAttribute("twin:faultProbability").Get(),
        "status": prim.GetAttribute("twin:status").Get(),
        "threshold": THRESHOLD,
    }


# ------------------------------------------------------------------ AI call --
async def infer(session, temperature: float, vibration: float) -> float:
    payload = {"inputs": [{"name": "input_0", "shape": [1, 2],
                           "datatype": "FP32", "data": [temperature, vibration]}]}
    async with session.post(INFER_URL, json=payload) as r:
        r.raise_for_status()
        body = await r.json()
    return float(body["outputs"][0]["data"][0])


# ------------------------------------------------------------- main loop ----
async def twin_loop(app):
    stage = app["stage"]
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.ws_connect(SENSOR_URL) as ws:
                    print("[twin] connected to sensor")
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        reading = json.loads(msg.data)
                        prob = await infer(session, reading["temperature"],
                                           reading["vibration"])
                        update_stage(stage, reading, prob)
                        state = read_stage(stage)
                        print(f"[twin] T={state['temperature']:.1f} "
                              f"V={state['vibration']:.3f} "
                              f"p={prob:.2f} {state['status']}")
                        for v in list(viewers):
                            await v.send_str(json.dumps(state))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[twin] {e!r} - retrying in 3 s")
                await asyncio.sleep(3)


# ----------------------------------------------------------------- viewer ---
async def index(request):
    return web.FileResponse(Path(__file__).with_name("viewer.html"))


async def viewer_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    viewers.add(ws)
    await ws.send_str(json.dumps(read_stage(request.app["stage"])))
    try:
        async for _ in ws:
            pass
    finally:
        viewers.discard(ws)
    return ws


async def on_start(app):
    app["stage"] = build_stage()
    app["loop"] = asyncio.create_task(twin_loop(app))
    print(f"[twin] stage: {STAGE_FILE.resolve()}")
    print("[twin] viewer: http://localhost:8080")


async def on_stop(app):
    app["loop"].cancel()


app = web.Application()
app.add_routes([web.get("/", index), web.get("/ws", viewer_ws)])
app.on_startup.append(on_start)
app.on_cleanup.append(on_stop)

if __name__ == "__main__":
    web.run_app(app, host="localhost", port=8080, print=None)

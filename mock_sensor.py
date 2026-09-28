"""
Mock IoT sensor: a WebSocket server that streams one reading per second.

Every 30 s the machine has a 10 s "overheating" episode, so the AI
prediction (and the twin's colour) visibly changes.

    python mock_sensor.py               -> ws://localhost:8765
"""
import asyncio
import json
import random
import time

from aiohttp import web, WSMsgType

START = time.time()


def reading() -> dict:
    t = time.time() - START
    faulty = (t % 30) > 20
    if faulty:
        temp, vib = random.uniform(86, 93), random.uniform(0.44, 0.55)
    else:
        temp, vib = random.uniform(62, 78), random.uniform(0.12, 0.32)
    return {"machine_id": "Machine_1", "timestamp": round(time.time(), 2),
            "temperature": round(temp, 2), "vibration": round(vib, 3)}


async def stream(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print("client connected")
    try:
        while not ws.closed:
            data = reading()
            await ws.send_str(json.dumps(data))
            print(data)
            await asyncio.sleep(1)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    print("client disconnected")
    return ws


app = web.Application()
app.add_routes([web.get("/", stream)])

if __name__ == "__main__":
    print("sensor streaming on ws://localhost:8765")
    web.run_app(app, host="localhost", port=8765, print=None)

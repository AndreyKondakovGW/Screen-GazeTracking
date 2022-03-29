from fastapi import WebSocket, WebSocketDisconnect, APIRouter
import pybase64
import asyncio

import app.services.launcher as launcher


router = APIRouter()


@router.websocket("/")
async def stream(websocket: WebSocket):
    launcher.run()
    await websocket.accept()
    try:
        while True:
            if not launcher.q_output.empty():
                frame = pybase64.b64encode_as_string(launcher.q_output.get_nowait())
                await websocket.send_text(f"data:image/jpeg;base64, {frame}")
                try:
                    await asyncio.wait_for(websocket.receive(), 0.01)
                except asyncio.TimeoutError:
                    ...
    except RuntimeError:
        launcher.stop()







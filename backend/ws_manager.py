import json
from typing import List

from fastapi import WebSocket


class WSManager:
    def __init__(self) -> None:
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, event_type: str, payload: dict) -> None:
        msg = json.dumps({"type": event_type, "payload": payload})
        stale = []
        for ws in self.connections:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

manager = WSManager()

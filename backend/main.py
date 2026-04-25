import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import backend.api as api_module
from backend.api import router
from backend.db import init_db
from backend.ws_manager import manager as ws_manager

app = FastAPI(title="ATLAS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep existing frontend contract under `/api/*`.
app.include_router(router, prefix="/api")
# Also expose OpenEnv-style endpoints at the root (`/reset`, `/step`, `/state`),
# so OpenEnv clients/judges can interact without needing the `/api` prefix.
app.include_router(router)


@app.on_event("startup")
def startup():
    init_db()
    sim_service = api_module.ensure_sim()
    if sim_service.llm.is_enabled():
        print("--- ATLAS AI Mode Active: LLM CEO is taking charge! ---")
    else:
        print("--- ATLAS Manual/Random Mode: No API keys found. ---")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            # Re-read the current sim each tick so Reset switches the live loop immediately.
            sim = api_module.ensure_sim()
            if sim.auto_play:
                frame = sim.step()
                await ws_manager.broadcast("state_update", frame)
                if frame["event"]:
                    await ws_manager.broadcast("market_event", {"event": frame["event"]["name"]})
                await ws_manager.broadcast("reward_update", {"reward": frame["reward"]})
                if frame["done"]:
                    await ws_manager.broadcast("episode_done", {"final_state": frame["state"]})
                    # Keep websocket alive across episodes so UI remains connected
                    # after reset/restart. Auto-play pauses until user restarts.
                    sim.auto_play = False
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

# Mount frontend static files if they exist (for production deployment)
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"msg": "Frontend not built. Run 'npm run build' in frontend directory."}

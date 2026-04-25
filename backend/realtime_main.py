import asyncio
import logging
import os
import sys

# Load environment variables from .env for security before other imports
try:
    from dotenv import load_dotenv
    # Find the root .env file and load it
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from fastapi import FastAPI
import uvicorn

# Ensure the root directory is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.api import router
from backend.db import init_db
from backend.services.broker import broker
from backend.ws_manager import manager as ws_manager
from env.realtime_engine import RealtimeSimulationEngine
from agents.async_employee import AsyncEmployeeAgent
from backend.synthetic_data import SyntheticDataStreamer

logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(name)s - %(message)s")
logger = logging.getLogger("realtime_main")

app = FastAPI(title="ATLAS Realtime Enterprise API")
app.include_router(router, prefix="/api")

# Global instances
engine: RealtimeSimulationEngine = None
agents = []
streamer: SyntheticDataStreamer = None

async def _stream_state_updates(q: asyncio.Queue):
    while True:
        try:
            res = await q.get()
            await ws_manager.broadcast("state_update", res)
            q.task_done()
        except asyncio.CancelledError:
            break

async def _stream_action_feedback(q: asyncio.Queue):
    while True:
        try:
            res = await q.get()
            decision_payload = {
                "day": engine.env.day,
                "phase": ["morning", "afternoon", "evening"][engine.env.phase_idx],
                "action": res.get("action"),
                "rationale": res.get("rationale", res.get("actor", "System")),
                "reward": res.get("reward"),
                "state": engine.env.state.copy(),
                "episode_id": getattr(engine.env, "episode_id", 0)
            }
            await ws_manager.broadcast("state_update", decision_payload)
            q.task_done()
        except asyncio.CancelledError:
            break

async def _stream_market_events(q: asyncio.Queue):
    while True:
        try:
            res = await q.get()
            event_payload = {"event": {"name": res.get("title")}}
            await ws_manager.broadcast("market_event", event_payload)
            q.task_done()
        except asyncio.CancelledError:
            break

async def ws_bridge():
    """Bridges internal Pub/Sub broker events to the external WebSocket manager."""
    state_q = await broker.subscribe("state_update")
    action_q = await broker.subscribe("action_feedback")
    market_q = await broker.subscribe("market_events")
    
    t1 = asyncio.create_task(_stream_state_updates(state_q))
    t2 = asyncio.create_task(_stream_action_feedback(action_q))
    t3 = asyncio.create_task(_stream_market_events(market_q))
    
    try:
        await asyncio.gather(t1, t2, t3)
    except asyncio.CancelledError:
        t1.cancel()
        t2.cancel()
        t3.cancel()
    finally:
        await broker.unsubscribe("state_update", state_q)
        await broker.unsubscribe("action_feedback", action_q)
        await broker.unsubscribe("market_events", market_q)

@app.on_event("startup")
async def on_startup():
    init_db()
    await broker.connect()
    
    global engine, agents, streamer
    # 1. Start Engine (1 sec per day)
    engine = RealtimeSimulationEngine(preset="startup", day_duration_seconds=1.0)
    await engine.start()
    
    # 2. Start Agents
    agents = [
        AsyncEmployeeAgent(role="engineering", personality="Logical and focused on product stability."),
        AsyncEmployeeAgent(role="sales", personality="Aggressive growth and client acquisition."),
        AsyncEmployeeAgent(role="hr", personality="Empathetic, focuses on employee morale.")
    ]
    for agent in agents:
        await agent.start()
        
    # 3. Start Synthetic Streamer (every 5 seconds)
    streamer = SyntheticDataStreamer(interval_seconds=5.0)
    await streamer.start()
    
    # 4. Start WebSocket Bridge
    asyncio.create_task(ws_bridge())
    
    logger.info("Real-Time Enterprise Architecture initialized.")

@app.on_event("shutdown")
async def on_shutdown():
    global engine, agents, streamer
    if streamer: await streamer.stop()
    for agent in agents: await agent.stop()
    if engine: await engine.stop()
    await broker.close()
    logger.info("Real-Time Enterprise Architecture shutdown.")

if __name__ == "__main__":
    uvicorn.run("backend.realtime_main:app", host="127.0.0.1", port=8000, reload=True)

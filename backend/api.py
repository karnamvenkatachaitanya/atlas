import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import numpy as np

from backend.db import EpisodeLog, SessionLocal, StepLog
from backend.schemas import ResetRequest, StepRequest, ToolCallRequest, FeedbackRequest
from backend.services.report import generate_investor_report
from backend.services.simulator import SimulationService
from backend.tools import call_tool, list_tools
from backend.ws_manager import manager as ws_manager
from pydantic import BaseModel
from env.startup_env import AtlasOpenEnv

class ToggleAutoRequest(BaseModel):
    auto_play: bool

class TrainRequest(BaseModel):
    episodes: int = 16
    algorithm: str = "ppo"

router = APIRouter()
sim = None


def ensure_sim() -> SimulationService:
    global sim
    if sim is None:
        sim = SimulationService()
    return sim


@router.post("/reset")
def reset(req: ResetRequest):
    global sim
    preset = req.preset
    if preset == "procedural":
        # Procedural mode randomizes across built-in presets.
        import random

        preset = random.choice(["startup", "growth", "crisis"])
    if preset not in {"startup", "growth", "crisis"}:
        raise HTTPException(status_code=400, detail=f"Unsupported preset: {req.preset}")

    sim = SimulationService(preset=preset)
    return {"ok": True, "state": sim.env.state, "episode_id": sim.episode_id}


@router.post("/step")
async def step(req: StepRequest):
    sim_obj = ensure_sim()
    frame = sim_obj.step(req.action_idx)
    
    # Broadcast to WS since we took a manual action
    await ws_manager.broadcast("state_update", frame)
    if frame["event"]:
        await ws_manager.broadcast("market_event", {"event": frame["event"]["name"]})
    await ws_manager.broadcast("reward_update", {"reward": frame["reward"]})
    if frame["done"]:
        await ws_manager.broadcast("episode_done", {"final_state": frame["state"]})
        
    return frame

@router.post("/toggle-auto")
def toggle_auto(req: ToggleAutoRequest):
    sim_obj = ensure_sim()
    sim_obj.auto_play = req.auto_play
    return {"ok": True, "auto_play": sim_obj.auto_play}


@router.get("/state")
def state():
    current_sim = ensure_sim()
    return {
        "state": current_sim.env.state,
        "done": current_sim.done,
        "log_size": len(current_sim.decision_log),
        "episode_id": current_sim.episode_id,
        "rl_metrics": getattr(current_sim, "rl_metrics", {}),
    }


@router.post("/train")
async def train_rl_model(req: TrainRequest):
    """Trigger RL training episodes and return metrics."""
    sim_obj = ensure_sim()
    
    # Run training episodes
    training_rewards = []
    episode_details = []
    
    for ep in range(req.episodes):
        env = AtlasOpenEnv(preset=sim_obj.preset)
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < 270:  # max_steps = 90 days * 3 phases
            # Use current policy (random for now, will be replaced by trained model)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        training_rewards.append(total_reward)
        episode_details.append({
            "episode": ep + 1,
            "reward": total_reward,
            "steps": steps
        })
    
    # Compute training metrics
    rewards_array = np.array(training_rewards)
    metrics = {
        "mean_reward": float(np.mean(rewards_array)),
        "std_reward": float(np.std(rewards_array)),
        "max_reward": float(np.max(rewards_array)),
        "min_reward": float(np.min(rewards_array)),
        "episode_rewards": training_rewards,
        "episode_details": episode_details,
        "algorithm": req.algorithm,
        "episodes_trained": req.episodes
    }
    
    # Store in simulator for dashboard access
    sim_obj.rl_metrics = {
        "episode_rewards": training_rewards,
        "step_rewards": [],
        "training_in_progress": False,
        "last_training_metrics": metrics
    }
    
    return metrics


@router.get("/tools/list")
def tools_list():
    return {"tools": list_tools()}


@router.post("/tools/call")
def tools_call(req: ToolCallRequest):
    try:
        return {"ok": True, "result": call_tool(req.tool, req.args)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/leaderboard")
def leaderboard(limit: int = 20):
    db = SessionLocal()
    rows = db.query(EpisodeLog).order_by(EpisodeLog.total_reward.desc()).limit(limit).all()
    out = [
        {
            "id": row.id,
            "mode": row.mode,
            "policy_name": row.policy_name,
            "total_reward": row.total_reward,
            "steps": row.steps,
            "created_at": row.created_at.isoformat() + "Z",
            "final_cash": row.final_cash,
            "final_revenue": row.final_revenue,
            "summary": row.summary or {},
        }
        for row in rows
    ]
    db.close()
    return out


@router.get("/replay/{episode_id}")
def replay_episode(episode_id: int):
    db = SessionLocal()
    steps = (
        db.query(StepLog)
        .filter(StepLog.episode_id == episode_id)
        .order_by(StepLog.id.asc())
        .all()
    )
    db.close()
    if not steps:
        raise HTTPException(status_code=404, detail="Episode not found")
    return [
        {
            "day": step.day,
            "phase": step.phase,
            "action": step.action,
            "reward": step.reward,
            "event": step.event,
            "state": step.state,
        }
        for step in steps
    ]


@router.get("/investor-report/{episode_id}")
def investor_report(episode_id: int):
    db = SessionLocal()
    ep = db.query(EpisodeLog).filter(EpisodeLog.id == episode_id).first()
    db.close()
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")

    os.makedirs("data", exist_ok=True)
    path = f"data/investor_report_{episode_id}.pdf"
    generate_investor_report(
        path,
        {
            "episode_id": ep.id,
            "mode": ep.mode,
            "policy": ep.policy_name,
            "total_reward": round(ep.total_reward, 2),
            "final_cash": round(ep.final_cash, 2),
            "final_revenue": round(ep.final_revenue, 2),
            "steps": ep.steps,
        },
    )
    return FileResponse(path=path, filename=os.path.basename(path), media_type="application/pdf")

@router.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    import json
    os.makedirs("data", exist_ok=True)
    feedback_file = "data/human_preferences.json"
    
    # Simple append to JSONL for easy dataset creation
    entry = {
        "episode_id": req.episode_id,
        "day": req.day,
        "phase": req.phase,
        "action": req.action,
        "state": req.state,
        "preference": req.preference
    }
    
    try:
        with open(feedback_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return {"ok": True, "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

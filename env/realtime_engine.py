import asyncio
import logging
from typing import Dict, Any, Optional

from env.startup_env import AtlasStartupEnv, ACTIONS
from backend.services.broker import broker

logger = logging.getLogger(__name__)

class RealtimeSimulationEngine:
    """
    An asynchronous wrapper around AtlasStartupEnv.
    Drives the simulation forward on a continuous real-time clock.
    Listens to 'agent_actions' topic to apply mutations.
    Publishes 'state_update' topic continuously.
    """
    def __init__(self, preset: str = "startup", day_duration_seconds: float = 1.0):
        self.env = AtlasStartupEnv(preset=preset)
        self.obs, self.info = self.env.reset()
        self.day_duration_seconds = day_duration_seconds
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._action_queue: Optional[asyncio.Queue] = None

    async def start(self):
        if self.running:
            return
        self.running = True
        self._action_queue = broker.subscribe("agent_actions")
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Realtime engine started. 1 day = {self.day_duration_seconds}s")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._action_queue:
            broker.unsubscribe("agent_actions", self._action_queue)
        logger.info("Realtime engine stopped.")

    async def _run_loop(self):
        phase_duration = self.day_duration_seconds / 3.0
        
        while self.running:
            # Check for any queued actions from agents/CEO before ticking time
            while not self._action_queue.empty():
                action_msg = await self._action_queue.get()
                self._apply_async_action(action_msg)
                self._action_queue.task_done()

            # The environment step in startup_env advances phase.
            # If no action was taken, we still want time to pass. We can pass a None or no-op action.
            # But startup_env requires an action. We'll pass an invalid action to just advance time,
            # or we need to modify it. Wait, startup_env penalizes invalid actions (-8 reward).
            # Instead of env.step(), we can just mutate state manually or use a "no_op" action.
            # Let's see if there is a no-op. If not, we will just advance time manually and calculate reward.
            
            # Since startup_env step() is coupled with CEO action, for realtime we will
            # trigger an empty step by default (simulate the world moving).
            # Actually, let's just broadcast the state and tick time.
            
            # Advance phase/day
            self.env.phase_idx += 1
            if self.env.phase_idx >= 3:
                self.env.phase_idx = 0
                self.env.day += 1

            # Natural burn rate & revenue application over time
            self.env.state["cash_balance"] += (self.env.state["revenue"] / 90.0) - (self.env.state["burn_rate"] / 90.0)
            self.env.state["cash_balance"] = max(0, self.env.state["cash_balance"])
            self.env._sanitize_state()

            # Publish state
            payload = {
                "day": self.env.day,
                "phase": ["morning", "afternoon", "evening"][self.env.phase_idx],
                "state": self.env.state.copy(),
                "done": self.env.day > self.env.max_days or self.env.state["cash_balance"] <= 0
            }
            await broker.publish("state_update", payload)

            if payload["done"]:
                await broker.publish("episode_done", {"final_state": self.env.state})
                self.running = False
                break

            await asyncio.sleep(phase_duration)

    def _apply_async_action(self, action_msg: Dict):
        """Apply an asynchronous action from an agent or CEO."""
        action_name = action_msg.get("action")
        intensity = action_msg.get("intensity", 1.0)
        actor = action_msg.get("actor", "ceo")
        
        if action_name in ACTIONS:
            # We call the internal _apply_action to mutate state without advancing the day clock
            reward, breakdown = self.env._apply_action(action_name, intensity)
            logger.info(f"Action applied asynchronously: {actor} did {action_name} (intensity={intensity}) -> reward={reward}")
            
            # Immediately publish state so UI updates instantly
            asyncio.create_task(broker.publish("action_feedback", {
                "actor": actor,
                "action": action_name,
                "reward": reward,
                "breakdown": breakdown
            }))

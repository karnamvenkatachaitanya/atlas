import asyncio
import json
import logging
import os
import random
from typing import Dict, Any, Optional

from backend.services.broker import broker
from env.startup_env import ACTIONS

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
HF_MODEL_URL = os.environ.get(
    "ATLAS_HF_API_URL", 
    "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"
)

class AsyncEmployeeAgent:
    """
    An independent microservice-like agent that reacts to the realtime state stream.
    Uses Hugging Face Inference APIs asynchronously to avoid blocking the event loop.
    """
    def __init__(self, role: str, personality: str):
        self.role = role
        self.personality = personality
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._state_queue: Optional[asyncio.Queue] = None

    async def start(self):
        if self.running: return
        self.running = True
        self._state_queue = await broker.subscribe("state_update")
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Agent [{self.role}] started listening for state updates.")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._state_queue:
            await broker.unsubscribe("state_update", self._state_queue)
        logger.info(f"Agent [{self.role}] stopped.")

    async def _run_loop(self):
        try:
            while self.running:
                payload = await self._state_queue.get()
                state = payload.get("state", {})
                
                # Agents don't react to every single tick, they act based on their own internal delays
                # e.g., Sales reacts if revenue drops, Engineering reacts to crises
                if self._should_react(state):
                    action_decision = await self._decide_action_async(state)
                    if action_decision:
                        await broker.publish("agent_actions", {
                            "actor": self.role,
                            "action": action_decision["action"],
                            "intensity": action_decision.get("intensity", 1.0),
                            "rationale": action_decision.get("rationale", "")
                        })
                self._state_queue.task_done()
        except asyncio.CancelledError:
            pass

    def _should_react(self, state: Dict[str, Any]) -> bool:
        """Determines if the agent should intervene based on state metrics."""
        # Add random jitter so they don't all act at once
        if random.random() > 0.3:
            return False

        if self.role == "engineering" and (state.get("crises", 0) > 0 or state.get("product_progress", 0) < 50):
            return True
        if self.role == "sales" and state.get("revenue", 0) < 10000:
            return True
        if self.role == "hr" and state.get("employee_morale", 0) < 50:
            return True
            
        return False

    async def _decide_action_async(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Makes an asynchronous call to Hugging Face API or uses local heuristics fallback."""
        prompt = (
            f"You are the {self.role.upper()} of a startup.\n"
            f"Personality: {self.personality}\n"
            f"Current State: Cash={state.get('cash_balance', 0):.0f}, Revenue={state.get('revenue', 0):.0f}, "
            f"Morale={state.get('employee_morale', 0):.1f}, Progress={state.get('product_progress', 0):.1f}, "
            f"Crises={state.get('crises', 0)}\n\n"
            "Choose one action to help the company from this list:\n"
            + ", ".join(ACTIONS) + "\n\n"
            "Format your response EXACTLY as JSON:\n"
            '{"action": "action_name", "intensity": 0.8, "rationale": "I chose this because..."}'
        )

        if HAS_AIOHTTP and HF_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 150, "temperature": 0.7, "return_full_text": False}
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(HF_MODEL_URL, headers=headers, json=payload, timeout=5.0) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            gen_text = data[0]["generated_text"]
                            # simple extraction
                            start = gen_text.find("{")
                            end = gen_text.rfind("}")
                            if start != -1 and end != -1:
                                return json.loads(gen_text[start:end+1])
            except Exception as e:
                logger.warning(f"[{self.role}] Async API call failed: {e}. Falling back to heuristic.")
        
        # Fallback to local heuristic if API fails or is not configured
        await asyncio.sleep(0.5) # Simulate thought delay
        
        if self.role == "engineering":
            action = "fix_bug_crisis" if state.get("crises", 0) > 0 else "assign_engineering_task"
        elif self.role == "sales":
            action = "run_ads" if state.get("cash_balance", 0) > 20000 else "negotiate_client"
        elif self.role == "hr":
            action = "improve_culture"
        else:
            action = random.choice(ACTIONS)

        return {
            "action": action,
            "intensity": 0.8 + (random.random() * 0.2),
            "rationale": f"As {self.role}, my local heuristic decided this based on current metrics."
        }

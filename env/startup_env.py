from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# OpenEnv packaging note:
# - The latest releases are distributed as `openenv-core` and expose `openenv.core.*`.
# - Some older examples use `openenv.env.Env`.
# We keep compatibility with both so Colab installs of `openenv-core` don't fail.
from openenv.core import Environment as OpenEnvBase

from env.events import maybe_event
from env.presets import PRESETS

MANDATES = [
    "Maximize Growth: Prioritize product progress and revenue even if burn rate increases.",
    "Cost Efficiency: Minimize burn rate and preserve cash balance at all costs.",
    "Balanced Stability: Maintain a healthy balance between employee morale and revenue.",
]

ACTIONS = [
    "hire_employee",
    "fire_employee",
    "increase_salaries",
    "assign_engineering_task",
    "launch_product",
    "run_ads",
    "negotiate_client",
    "reduce_costs",
    "raise_funding",
    "fix_bug_crisis",
    "improve_culture",
    "give_bonuses",
    "change_roadmap",
]

PHASES = ["morning", "afternoon", "evening"]


class AtlasStartupEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, preset: str = "startup"):
        super().__init__()
        self.preset = preset
        self.max_days = 90
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=2_000_000, shape=(10,), dtype=np.float32
        )
        self.day = 1
        self.phase_idx = 0
        self.state: Dict[str, float] = {}
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        import random
        cfg = PRESETS[self.preset]
        self.day = 1
        self.phase_idx = 0
        self.mandate = options.get("mandate") if options else None
        if not self.mandate:
            self.mandate = random.choice(MANDATES)

        self.state = {
            "cash_balance": float(cfg["cash"]),
            "revenue": float(cfg["revenue"]),
            "burn_rate": float(cfg["burn_rate"]),
            "employee_morale": 70.0,
            "product_progress": 20.0,
            "customer_satisfaction": 65.0,
            "investor_trust": float(cfg["investor_trust"]),
            "pending_tasks": 5.0,
            "crises": 0.0,
            "market_trend": 0.0,
        }
        self._sanitize_state()
        return self._obs(), {"phase": PHASES[self.phase_idx], "day": self.day, "mandate": self.mandate}

    def observation(self) -> np.ndarray:
        """Public observation accessor for training/evaluation code."""
        return self._obs()

    def state_snapshot(self) -> Dict[str, float]:
        """Stable read-only state view to avoid accidental in-place mutation by callers."""
        return dict(self.state)

    def _obs(self) -> np.ndarray:
        s = self.state
        return np.array(
            [
                s["cash_balance"],
                s["revenue"],
                s["burn_rate"],
                s["employee_morale"],
                s["product_progress"],
                s["customer_satisfaction"],
                s["investor_trust"],
                s["pending_tasks"],
                s["crises"],
                s["market_trend"],
            ],
            dtype=np.float32,
        )

    def step(self, action: int):
        invalid_action = not isinstance(action, (int, np.integer)) or not (0 <= int(action) < len(ACTIONS))
        action_name = "invalid_action" if invalid_action else ACTIONS[int(action)]
        event = maybe_event()
        if invalid_action:
            action_reward = -8.0
            action_reward_breakdown = {
                "action_reward": -8.0,
                "business_reward": 0.0,
                "revenue_reward": 0.0,
                "morale_reward": 0.0,
                "customer_reward": 0.0,
                "trust_reward": 0.0,
                "burn_penalty": 0.0,
                "crisis_penalty": 0.0,
            }
        else:
            action_reward, action_reward_breakdown = self._apply_action(action_name)
        reward = action_reward
        if event:
            event_reward = self._apply_event(event)
        else:
            event_reward = 0.0

        reward_breakdown = {
            **action_reward_breakdown,
            "event_reward": float(event_reward),
            "invalid_action_penalty": -8.0 if invalid_action else 0.0,
        }
        reward += event_reward

        self.state["cash_balance"] += self.state["revenue"] - self.state["burn_rate"] / 3
        self.state["cash_balance"] = max(0, self.state["cash_balance"])
        self._sanitize_state()

        finite_ok = all(np.isfinite(v) for v in self.state.values())
        if not finite_ok:
            reward -= 15.0
            reward_breakdown["finite_state_penalty"] = -15.0
        else:
            reward_breakdown["finite_state_penalty"] = 0.0

        self.phase_idx += 1
        if self.phase_idx >= 3:
            self.phase_idx = 0
            self.day += 1

        terminated = self.day > self.max_days or self.state["cash_balance"] <= 0 or not finite_ok
        truncated = False
        info = {
            "day": self.day,
            "phase": PHASES[self.phase_idx],
            "event": event,
            "action_name": action_name,
            "reward": reward,
            "invalid_action": bool(invalid_action),
            "reward_breakdown": reward_breakdown,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def _sanitize_state(self) -> None:
        # Clamp all mutable metrics to prevent runaway values and reward hacking.
        self.state["cash_balance"] = float(np.clip(self.state["cash_balance"], 0.0, 2_000_000.0))
        self.state["revenue"] = float(np.clip(self.state["revenue"], 0.0, 2_000_000.0))
        self.state["burn_rate"] = float(np.clip(self.state["burn_rate"], 0.0, 2_000_000.0))
        self.state["employee_morale"] = float(np.clip(self.state["employee_morale"], 0.0, 100.0))
        self.state["product_progress"] = float(np.clip(self.state["product_progress"], 0.0, 100.0))
        self.state["customer_satisfaction"] = float(np.clip(self.state["customer_satisfaction"], 0.0, 100.0))
        self.state["investor_trust"] = float(np.clip(self.state["investor_trust"], 0.0, 100.0))
        self.state["pending_tasks"] = float(np.clip(self.state["pending_tasks"], 0.0, 100.0))
        self.state["crises"] = float(np.clip(self.state["crises"], 0.0, 20.0))
        self.state["market_trend"] = float(np.clip(self.state["market_trend"], -100.0, 100.0))

    def _apply_action(self, action: str) -> tuple[float, Dict[str, float]]:
        reward_breakdown: Dict[str, float] = {
            "action_reward": 0.0,
            "business_reward": 0.0,
            "revenue_reward": 0.0,
            "morale_reward": 0.0,
            "customer_reward": 0.0,
            "trust_reward": 0.0,
            "burn_penalty": 0.0,
            "crisis_penalty": 0.0,
        }
        if action == "hire_employee":
            self.state["burn_rate"] += 2000
            self.state["product_progress"] += 2
            self.state["employee_morale"] += 1
        elif action == "fire_employee":
            self.state["burn_rate"] -= 1800
            self.state["employee_morale"] -= 5
        elif action == "increase_salaries":
            self.state["burn_rate"] += 3000
            self.state["employee_morale"] += 4
        elif action == "assign_engineering_task":
            self.state["product_progress"] += 3
            self.state["pending_tasks"] = max(0, self.state["pending_tasks"] - 1)
        elif action == "launch_product":
            self.state["revenue"] += 7000
            self.state["product_progress"] -= 5
            reward_breakdown["action_reward"] += 8.0
        elif action == "run_ads":
            self.state["burn_rate"] += 2500
            self.state["revenue"] += 3000
        elif action == "negotiate_client":
            self.state["revenue"] += 5000
            self.state["investor_trust"] += 1
        elif action == "reduce_costs":
            self.state["burn_rate"] -= 2500
            self.state["employee_morale"] -= 2
        elif action == "raise_funding":
            self.state["cash_balance"] += 120000
            self.state["investor_trust"] -= 2
        elif action == "fix_bug_crisis":
            self.state["customer_satisfaction"] += 3
            self.state["crises"] = max(0, self.state["crises"] - 1)
        elif action == "improve_culture":
            self.state["employee_morale"] += 4
            self.state["burn_rate"] += 1200
        elif action == "give_bonuses":
            self.state["cash_balance"] -= 10000
            self.state["employee_morale"] += 5
        elif action == "change_roadmap":
            self.state["product_progress"] += 1
            self.state["pending_tasks"] += 1

        reward_breakdown["revenue_reward"] = 0.00005 * self.state["revenue"]
        reward_breakdown["morale_reward"] = 0.02 * self.state["employee_morale"]
        reward_breakdown["customer_reward"] = 0.02 * self.state["customer_satisfaction"]
        reward_breakdown["trust_reward"] = 0.01 * self.state["investor_trust"]
        reward_breakdown["burn_penalty"] = -0.00004 * self.state["burn_rate"]
        reward_breakdown["crisis_penalty"] = -0.02 * self.state["crises"]
        reward_breakdown["business_reward"] = (
            reward_breakdown["revenue_reward"]
            + reward_breakdown["morale_reward"]
            + reward_breakdown["customer_reward"]
            + reward_breakdown["trust_reward"]
            + reward_breakdown["burn_penalty"]
            + reward_breakdown["crisis_penalty"]
        )
        reward_breakdown["action_reward"] += reward_breakdown["business_reward"]
        return float(reward_breakdown["action_reward"]), reward_breakdown

    def _apply_event(self, event: str) -> float:
        if event == "server_outage":
            self.state["customer_satisfaction"] -= 8
            self.state["crises"] += 1
            return -5.0
        if event == "market_crash":
            self.state["revenue"] *= 0.85
            self.state["investor_trust"] -= 6
            return -6.0
        if event == "viral_growth":
            self.state["revenue"] *= 1.25
            self.state["customer_satisfaction"] += 3
            return 8.0
        if event == "key_employee_resigns":
            self.state["product_progress"] -= 4
            self.state["employee_morale"] -= 6
            return -7.0
        if event == "customer_complaints_spike":
            self.state["customer_satisfaction"] -= 10
            self.state["crises"] += 1
            return -6.0
        return -1.0 if "risk" in event or "delayed" in event else 1.0

    def render(self):
        return f"Day {self.day} {PHASES[self.phase_idx]} :: {self.state}"


class AtlasOpenEnv(OpenEnvBase):
    """
    Explicit OpenEnv adapter for AtlasStartupEnv.
    This keeps the same simulation dynamics while exposing OpenEnv's base Env usage.
    """

    def __init__(self, preset: str = "startup"):
        self.core = AtlasStartupEnv(preset=preset)
        super().__init__()
        # Keep Gym-style aliases for compatibility with existing tooling.
        self.observation_space = self.core.observation_space
        self.action_space = self.core.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.core.reset(seed=seed, options=options)
        self.mandate = info["mandate"]
        return obs, info

    def step(self, action: int):
        return self.core.step(action)

    def observation(self):
        return self.core.observation()

    def state(self):
        s = self.core.state.copy()
        s["mandate"] = getattr(self, "mandate", "None")
        return s

    def render(self):
        return self.core.render()

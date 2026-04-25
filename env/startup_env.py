from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# OpenEnv packaging note:
# - The latest releases are distributed as `openenv-core` and expose `openenv.core.*`.
# - Some judge/dev environments may not have OpenEnv installed when running pure-env scripts.
# We therefore make OpenEnv optional at import time, while keeping full functionality
# when `openenv-core` is installed.
try:
    from openenv.core import Environment as OpenEnvBase
except Exception:  # pragma: no cover
    class OpenEnvBase:  # type: ignore
        pass

from env.events import maybe_event
from env.presets import PRESETS
from agents.employee import EmployeeAgent

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
        self.action_space = spaces.Dict({
            "action": spaces.Discrete(len(ACTIONS)),
            "intensity": spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)
        })
        # Normalized observation space: all values in [0, 1] for stable RL training
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )
        self.day = 1
        self.phase_idx = 0
        self.state: Dict[str, float] = {}
        self.inbox: str = ""
        self.agents: List[EmployeeAgent] = []
        self._action_counts: Dict[str, int] = {a: 0 for a in ACTIONS}
        self._last_action_idx: int = -1
        self._pending_event_chain: str = ""  # tracks cascading event chains
        self._mandate_id: int = 0
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        import random
        self.day = 1
        self.phase_idx = 0
        self.mandate = options.get("mandate") if options else None
        if not self.mandate:
            self.mandate = random.choice(MANDATES)
        # Encode mandate as integer for observation space.
        self._mandate_id = next((i for i, m in enumerate(MANDATES) if m == self.mandate), 0)

        if self.preset == "procedural":
            self.state = {
                "cash_balance": float(random.uniform(50_000, 500_000)),
                "revenue": float(random.uniform(0, 50_000)),
                "burn_rate": float(random.uniform(10_000, 50_000)),
                "employee_morale": float(random.uniform(40, 100)),
                "product_progress": float(random.uniform(0, 50)),
                "customer_satisfaction": float(random.uniform(40, 100)),
                "investor_trust": float(random.uniform(40, 100)),
                "pending_tasks": float(random.uniform(0, 20)),
                "crises": float(random.randint(0, 2)),
                "market_trend": float(random.uniform(-10, 10)),
            }
        else:
            cfg = PRESETS[self.preset]
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

        self.agents = [
            EmployeeAgent(role="engineering_manager", personality={"focus": "product"}),
            EmployeeAgent(role="sales_lead", personality={"focus": "revenue"}),
            EmployeeAgent(role="hr_recruiter", personality={"focus": "morale"}),
            EmployeeAgent(role="finance_officer", personality={"focus": "cash"}),
            EmployeeAgent(role="customer_success", personality={"focus": "csat"}),
        ]
        
        inbox_msgs = []
        for agent in self.agents:
            agent.propose_action(self.state)
        for agent in self.agents:
            neg_result = agent.negotiate(self.agents, self.state)
            inbox_msgs.append(neg_result)
        self.inbox = "\n".join(inbox_msgs)

        self._action_counts = {a: 0 for a in ACTIONS}
        self._last_action_idx = -1
        self._pending_event_chain = ""
        self._sanitize_state()
        return self._obs(), {"phase": PHASES[self.phase_idx], "day": self.day, "mandate": self.mandate}

    def observation(self) -> np.ndarray:
        """Public observation accessor for training/evaluation code."""
        return self._obs()

    def _normalize_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1] range for stable RL training."""
        normalized = raw_obs.copy()
        # State metrics (indices 0-9) - normalize by max values
        normalized[0] = np.clip(raw_obs[0] / 2_000_000, 0.0, 1.0)  # cash_balance
        normalized[1] = np.clip(raw_obs[1] / 2_000_000, 0.0, 1.0)  # revenue
        normalized[2] = np.clip(raw_obs[2] / 2_000_000, 0.0, 1.0)  # burn_rate
        normalized[3] = np.clip(raw_obs[3] / 100.0, 0.0, 1.0)      # employee_morale
        normalized[4] = np.clip(raw_obs[4] / 100.0, 0.0, 1.0)      # product_progress
        normalized[5] = np.clip(raw_obs[5] / 100.0, 0.0, 1.0)      # customer_satisfaction
        normalized[6] = np.clip(raw_obs[6] / 100.0, 0.0, 1.0)      # investor_trust
        normalized[7] = np.clip(raw_obs[7] / 100.0, 0.0, 1.0)      # pending_tasks
        normalized[8] = np.clip(raw_obs[8] / 20.0, 0.0, 1.0)       # crises
        normalized[9] = np.clip((raw_obs[9] + 100) / 200, 0.0, 1.0)  # market_trend [-100,100] -> [0,1]
        # Extended features (indices 10-13) already in [0,1] or small ints
        normalized[10] = np.clip(raw_obs[10], 0.0, 1.0)  # day_fraction [0-1]
        normalized[11] = np.clip(raw_obs[11], 0.0, 1.0)  # phase_fraction [0-1]
        normalized[12] = np.clip(raw_obs[12] / 2.0, 0.0, 1.0)  # mandate_id [0-2] -> [0-1]
        normalized[13] = np.clip((raw_obs[13] + 1) / 13.0, 0.0, 1.0)  # last_action [-1,12] -> [0,1]
        return normalized

    def state_snapshot(self) -> Dict[str, float]:
        """Stable read-only state view to avoid accidental in-place mutation by callers."""
        return dict(self.state)

    def _obs(self) -> np.ndarray:
        s = self.state
        raw_obs = np.array(
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
                # Extended context: lets agent reason about time, mandate, and history.
                float(self.day) / float(self.max_days),     # day fraction [0-1]
                float(self.phase_idx) / 2.0,                # phase fraction [0-1]
                float(self._mandate_id),                    # mandate encoding
                float(self._last_action_idx),               # last action taken (-1 if none)
            ],
            dtype=np.float32,
        )
        return self._normalize_obs(raw_obs)

    def _dept_priority(self) -> tuple[str, float, str]:
        """
        Lightweight negotiation signal:
        determine which department need is most urgent right now.
        Returns: (dept, severity, rationale)
        """
        s = self.state
        crises = float(s["crises"])
        csat = float(s["customer_satisfaction"])
        cash = float(s["cash_balance"])
        burn = float(s["burn_rate"])
        progress = float(s["product_progress"])
        morale = float(s["employee_morale"])

        finance_sev = max(0.0, (200_000.0 - cash) / 50_000.0) + max(0.0, (burn - 25_000.0) / 10_000.0)
        eng_sev = max(0.0, (70.0 - progress) / 10.0)
        cs_sev = max(0.0, (65.0 - csat) / 8.0) + crises * 0.8
        hr_sev = max(0.0, (60.0 - morale) / 8.0)
        sales_sev = max(0.0, (25_000.0 - float(s["revenue"])) / 10_000.0)

        scores = {
            "finance": finance_sev,
            "engineering": eng_sev,
            "customer_success": cs_sev,
            "hr": hr_sev,
            "sales": sales_sev,
        }
        dept = max(scores, key=scores.get)
        sev = float(scores[dept])
        rationale = f"{dept}_sev={sev:.2f}"
        return dept, sev, rationale

    def _preferred_action_idx(self) -> tuple[int, str]:
        """
        Map the most urgent department need into a suggested CEO action.
        This creates a measurable “negotiation alignment” signal without changing the action space.
        """
        dept, sev, rationale = self._dept_priority()
        if dept == "finance":
            suggested = "reduce_costs" if self.state["cash_balance"] < 300_000 else "negotiate_client"
        elif dept == "engineering":
            suggested = "assign_engineering_task" if self.state["product_progress"] < 80 else "launch_product"
        elif dept == "customer_success":
            suggested = "fix_bug_crisis"
        elif dept == "hr":
            suggested = "improve_culture"
        else:  # sales
            suggested = "negotiate_client" if self.state["burn_rate"] > 30_000 else "run_ads"

        return ACTIONS.index(suggested), f"{rationale} suggested={suggested} mandate={getattr(self,'mandate','')[:24]}"

    def step(self, action):
        if isinstance(action, dict):
            action_idx = int(action.get("action", 0))
            intensity = float(action.get("intensity", [1.0])[0]) if isinstance(action.get("intensity"), (list, np.ndarray)) else float(action.get("intensity", 1.0))
        else:
            action_idx = int(action)
            intensity = 1.0

        invalid_action = not isinstance(action_idx, (int, np.integer)) or not (0 <= int(action_idx) < len(ACTIONS))
        action_name = "invalid_action" if invalid_action else ACTIONS[int(action_idx)]
        event = maybe_event(self.state)
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
            action_reward, action_reward_breakdown = self._apply_action(action_name, intensity)
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

        # Agent reactions, autonomy, and next proposals
        inbox_msgs = []
        if not invalid_action:
            for agent in self.agents:
                reaction = agent.react(action_name, self.state)
                agent.execute_action(self.state)
                agent.propose_action(self.state)
            
            # Negotiate before finalizing inbox
            for agent in self.agents:
                neg_result = agent.negotiate(self.agents, self.state)
                # react returns a dict now, so we extract the message
                react_msg = agent.memory[-1]['message'] if agent.memory else ""
                inbox_msgs.append(f"{agent.role.upper()} ({agent.happiness:.0f}% morale): {react_msg} -> {neg_result}")
                
        self.inbox = "\n".join(inbox_msgs)

        # Department priority bonus (keep as shaping reward)
        preferred_idx, preferred_rationale = self._preferred_action_idx()
        if invalid_action:
            priority_bonus = 0.0
            priority_match = False
        else:
            priority_match = int(action) == int(preferred_idx)
            _dept, severity, _rat = self._dept_priority()
            scale = 0.6 if severity >= 2.0 else 0.2
            priority_bonus = (1.0 if priority_match else -0.3) * scale
        reward += float(priority_bonus)
        reward_breakdown["priority_bonus"] = float(priority_bonus)

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
            "inbox": self.inbox,
            "reward_breakdown": reward_breakdown,
            "priority": {
                "preferred_action_idx": int(preferred_idx),
                "preferred_action_name": ACTIONS[int(preferred_idx)],
                "matched": bool(priority_match),
                "rationale": preferred_rationale,
            },
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

    def _noise(self, base: float, scale: float = 0.15) -> float:
        """Add Gaussian noise to transition values — makes dynamics stochastic."""
        return base * (1.0 + np.random.normal(0.0, scale))

    def _diminishing(self, action: str, base: float) -> float:
        """Diminishing returns: repeated use of the same action is less effective."""
        count = self._action_counts.get(action, 0)
        # Exponential decay: effectiveness halves roughly every 16 uses.
        factor = max(0.2, 1.0 / (1.0 + count * 0.06))
        return base * factor

    def _apply_action(self, action: str, intensity: float = 1.0) -> tuple[float, Dict[str, float]]:
        reward_breakdown: Dict[str, float] = {
            "action_reward": 0.0,
            "business_reward": 0.0,
            "revenue_reward": 0.0,
            "morale_reward": 0.0,
            "customer_reward": 0.0,
            "trust_reward": 0.0,
            "burn_penalty": 0.0,
            "crisis_penalty": 0.0,
            "repetition_penalty": 0.0,
        }
        # Track action usage for diminishing returns and anti-gaming.
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        self._last_action_idx = ACTIONS.index(action)

        if action == "hire_employee":
            self.state["burn_rate"] += self._noise(self._diminishing(action, 2000 * intensity))
            self.state["product_progress"] += self._noise(self._diminishing(action, 2 * intensity))
            self.state["employee_morale"] += self._noise(self._diminishing(action, 1 * intensity))
        elif action == "fire_employee":
            self.state["burn_rate"] -= self._noise(1800 * intensity)
            # Morale hit is worse if morale is already low (non-linear interaction).
            morale_hit = 5 * intensity * (1.0 + max(0, 50 - self.state["employee_morale"]) / 50.0)
            self.state["employee_morale"] -= self._noise(morale_hit)
        elif action == "increase_salaries":
            self.state["burn_rate"] += self._noise(3000 * intensity)
            self.state["employee_morale"] += self._noise(self._diminishing(action, 4 * intensity))
        elif action == "assign_engineering_task":
            # Effectiveness depends on morale — demoralized teams are less productive.
            morale_factor = max(0.3, self.state["employee_morale"] / 70.0)
            self.state["product_progress"] += self._noise(3 * intensity * morale_factor)
            self.state["pending_tasks"] = max(0, self.state["pending_tasks"] - 1 * intensity)
        elif action == "launch_product":
            # Action-state interaction: launching with low progress is risky.
            if self.state["product_progress"] < 40:
                # Premature launch: reduced revenue, damages trust and CSAT.
                self.state["revenue"] += self._noise(2000 * intensity)
                self.state["customer_satisfaction"] -= self._noise(5 * intensity)
                self.state["investor_trust"] -= self._noise(3 * intensity)
                reward_breakdown["action_reward"] += 1.0
            else:
                self.state["revenue"] += self._noise(self._diminishing(action, 7000 * intensity))
                reward_breakdown["action_reward"] += 8.0
            self.state["product_progress"] -= self._noise(5 * intensity)
        elif action == "run_ads":
            self.state["burn_rate"] += self._noise(2500 * intensity)
            # Ad effectiveness depends on CSAT — bad product means wasted spend.
            csat_factor = max(0.2, self.state["customer_satisfaction"] / 65.0)
            self.state["revenue"] += self._noise(self._diminishing(action, 3000 * intensity * csat_factor))
        elif action == "negotiate_client":
            # Harder to negotiate with low trust.
            trust_factor = max(0.3, self.state["investor_trust"] / 60.0)
            self.state["revenue"] += self._noise(self._diminishing(action, 5000 * intensity * trust_factor))
            self.state["investor_trust"] += self._noise(self._diminishing(action, 1 * intensity))
        elif action == "reduce_costs":
            self.state["burn_rate"] -= self._noise(self._diminishing(action, 2500 * intensity))
            self.state["employee_morale"] -= self._noise(2 * intensity)
        elif action == "raise_funding":
            # Repeated fundraising erodes trust faster.
            count = self._action_counts.get(action, 1)
            self.state["cash_balance"] += self._noise(self._diminishing(action, 120000 * intensity))
            self.state["investor_trust"] -= self._noise((2 + count * 0.5) * intensity)
        elif action == "fix_bug_crisis":
            self.state["customer_satisfaction"] += self._noise(3 * intensity)
            self.state["crises"] = max(0, self.state["crises"] - 1 * intensity)
        elif action == "improve_culture":
            self.state["employee_morale"] += self._noise(self._diminishing(action, 4 * intensity))
            self.state["burn_rate"] += self._noise(1200 * intensity)
        elif action == "give_bonuses":
            self.state["cash_balance"] -= self._noise(10000 * intensity)
            self.state["employee_morale"] += self._noise(self._diminishing(action, 5 * intensity))
        elif action == "change_roadmap":
            self.state["product_progress"] += self._noise(1 * intensity)
            self.state["pending_tasks"] += self._noise(1 * intensity)

        # Anti-gaming: penalize excessive repetition of the same action.
        repeat_count = self._action_counts.get(action, 0)
        if repeat_count > 20:
            repetition_pen = -0.1 * (repeat_count - 20)
            reward_breakdown["repetition_penalty"] = repetition_pen
        else:
            reward_breakdown["repetition_penalty"] = 0.0

        # SIMPLIFIED REWARD - aligned with mandate for stable learning
        s = self.state
        mandate = getattr(self, 'mandate', '')
        
        # Base reward components (normalized to meaningful ranges)
        revenue_norm = (s["revenue"] - 15000) / 50000  # ~[-1, 1] centered around startup revenue
        cash_norm = min(s["cash_balance"] / 500000, 1.0)  # [0, 1] - healthy cash position
        morale_norm = (s["employee_morale"] - 50) / 50  # [-1, 1] - above/below neutral
        progress_norm = s["product_progress"] / 100  # [0, 1] - completion percentage
        
        # Mandate-specific rewards - clear signal for RL agent
        if "Maximize Growth" in mandate:
            # Growth-focused: prioritize progress and revenue
            reward = 0.4 * progress_norm + 0.4 * revenue_norm + 0.2 * cash_norm
        elif "Cost Efficiency" in mandate:
            # Efficiency-focused: preserve cash, control burn
            burn_norm = s["burn_rate"] / 50000  # normalize burn
            reward = 0.5 * cash_norm - 0.3 * burn_norm + 0.2 * morale_norm
        else:  # Balanced Stability
            # Balanced: all metrics matter equally
            reward = 0.25 * revenue_norm + 0.25 * morale_norm + 0.25 * cash_norm + 0.25 * progress_norm
        
        # Penalties for negative states
        crisis_penalty = -0.1 * s["crises"]
        repeat_penalty = reward_breakdown.get("repetition_penalty", 0.0)
        
        # Combine rewards
        reward += crisis_penalty + repeat_penalty
        
        # Update breakdown for transparency
        reward_breakdown["action_reward"] = reward
        reward_breakdown["business_reward"] = reward - crisis_penalty - repeat_penalty
        reward_breakdown["revenue_reward"] = 0.25 * revenue_norm
        reward_breakdown["morale_reward"] = 0.25 * morale_norm
        reward_breakdown["customer_reward"] = 0.0
        reward_breakdown["trust_reward"] = 0.0
        reward_breakdown["burn_penalty"] = crisis_penalty
        reward_breakdown["crisis_penalty"] = crisis_penalty
        
        return float(reward), reward_breakdown

    def _apply_event(self, event: str) -> float:
        # Event chains: some events trigger follow-up events if not addressed.
        if self._pending_event_chain:
            chain = self._pending_event_chain
            self._pending_event_chain = ""
            # Cascading crisis: previous event wasn't resolved.
            if chain == "server_outage" and self.state["crises"] > 0:
                self.state["customer_satisfaction"] -= self._noise(12)
                self.state["crises"] += 1
                self.state["investor_trust"] -= self._noise(4)
                # Note: still process the current event below.
            elif chain == "key_employee_resigns" and self.state["employee_morale"] < 50:
                # Low morale after resignation → another person quits (cascade).
                self.state["product_progress"] -= self._noise(3)
                self.state["employee_morale"] -= self._noise(4)
                self.state["burn_rate"] -= self._noise(1500)  # fewer employees

        if event == "server_outage":
            self.state["customer_satisfaction"] -= self._noise(8)
            self.state["crises"] += 1
            self._pending_event_chain = "server_outage"  # will cascade if not fixed
            return -5.0
        if event == "market_crash":
            severity = 0.75 + np.random.uniform(0, 0.15)  # 75-90% of revenue survives
            self.state["revenue"] *= severity
            self.state["investor_trust"] -= self._noise(6)
            return -6.0
        if event == "viral_growth":
            boost = 1.15 + np.random.uniform(0, 0.20)  # 115-135% revenue boost
            self.state["revenue"] *= boost
            self.state["customer_satisfaction"] += self._noise(3)
            return 8.0
        if event == "key_employee_resigns":
            self.state["product_progress"] -= self._noise(4)
            self.state["employee_morale"] -= self._noise(6)
            self._pending_event_chain = "key_employee_resigns"
            return -7.0
        if event == "customer_complaints_spike":
            self.state["customer_satisfaction"] -= self._noise(10)
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

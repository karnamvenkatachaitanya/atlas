import random
from typing import Dict

from agents.employee import EmployeeAgent
from agents.personalities import PERSONALITIES
from backend.db import EpisodeLog, SessionLocal, StepLog
from env.startup_env import ACTIONS, AtlasStartupEnv
from backend.services.llm_service import LLMService


class SimulationService:
    def __init__(self, preset: str = "startup", policy_name: str = "random"):
        self.preset = preset
        self.policy_name = policy_name
        self.env = AtlasStartupEnv(preset=preset)
        self.obs, self.info = self.env.reset()
        self.done = False
        self.decision_log = []
        self.total_reward = 0.0
        self.episode_id = None
        self.llm = LLMService()
        self.employee_agents = [
            EmployeeAgent("engineering_manager", PERSONALITIES["engineering_manager"]),
            EmployeeAgent("sales_lead", PERSONALITIES["sales_lead"]),
            EmployeeAgent("hr_recruiter", PERSONALITIES["hr_recruiter"]),
            EmployeeAgent("finance_officer", PERSONALITIES["finance_officer"]),
            EmployeeAgent("customer_success", PERSONALITIES["customer_success"]),
        ]
        self._start_episode()

    def _start_episode(self):
        db = SessionLocal()
        row = EpisodeLog(mode=self.preset, policy_name=self.policy_name)
        db.add(row)
        db.commit()
        db.refresh(row)
        self.episode_id = row.id
        db.close()

    def step(self, action_idx=None) -> Dict:
        if action_idx is None:
            if self.llm.is_enabled():
                action_idx = self.llm.get_action(self.env.state)
            else:
                action_idx = random.randint(0, len(ACTIONS) - 1)
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        self.obs = obs
        self.done = terminated or truncated
        self.total_reward += reward
        action_name = ACTIONS[action_idx]

        reactions = [agent.react(action_name, self.env.state) for agent in self.employee_agents]
        frame = {
            "state": self.env.state.copy(),
            "day": info["day"],
            "phase": info["phase"],
            "mandate": getattr(self.env, "mandate", "None"),
            "action": action_name,
            "reward": reward,
            "event": {"name": info.get("event")} if info.get("event") else None,
            "reactions": reactions,
            "done": self.done,
            "episode_id": self.episode_id,
        }
        self.decision_log.append(frame)
        self._persist_step(frame)
        if self.done:
            self._finalize_episode()
        return frame

    def _persist_step(self, frame: Dict) -> None:
        db = SessionLocal()
        db.add(
            StepLog(
                episode_id=self.episode_id,
                day=frame["day"],
                phase=frame["phase"],
                action=frame["action"],
                reward=float(frame["reward"]),
                event=frame["event"],
                state=frame["state"],
            )
        )
        db.commit()
        db.close()

    def _finalize_episode(self) -> None:
        db = SessionLocal()
        ep = db.query(EpisodeLog).filter(EpisodeLog.id == self.episode_id).first()
        if ep:
            ep.total_reward = float(self.total_reward)
            ep.steps = len(self.decision_log)
            ep.final_cash = float(self.env.state["cash_balance"])
            ep.final_revenue = float(self.env.state["revenue"])
            ep.summary = {
                "morale": self.env.state["employee_morale"],
                "customer_satisfaction": self.env.state["customer_satisfaction"],
                "investor_trust": self.env.state["investor_trust"],
            }
            db.commit()
        db.close()

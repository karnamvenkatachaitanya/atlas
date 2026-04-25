from typing import Any, Dict, Optional

from pydantic import BaseModel


class ResetRequest(BaseModel):
    preset: str = "startup"


class StepRequest(BaseModel):
    action_idx: Optional[int] = None

class FeedbackRequest(BaseModel):
    episode_id: int
    day: int
    phase: str
    action: str
    state: Dict[str, Any]
    preference: str  # 'upvote' or 'downvote'


class ToolCallRequest(BaseModel):
    tool: str
    args: Optional[Dict[str, Any]] = None


class ReplayStepOut(BaseModel):
    day: int
    phase: str
    action: str
    reward: float
    event: Optional[Dict[str, Any]]
    state: Dict[str, Any]

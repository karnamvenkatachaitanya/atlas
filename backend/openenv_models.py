from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AtlasAction(BaseModel):
    """
    Action model for OpenEnv manifest compatibility.
    
    Matches the 13 discrete actions in AtlasStartupEnv.
    """
    action_idx: int = Field(..., ge=0, le=12, description="Discrete action index (0-12)")
    preset: str = Field(default="startup", description="Environment preset")


class AtlasObservation(BaseModel):
    """
    Observation model for OpenEnv manifest compatibility.
    
    Captures the full 14-dimensional normalized observation space plus metadata.
    """
    # Core 10 state dimensions (normalized to [0,1])
    cash_balance: float = Field(..., ge=0, le=1, description="Normalized cash balance")
    revenue: float = Field(..., ge=0, le=1, description="Normalized revenue")
    burn_rate: float = Field(..., ge=0, le=1, description="Normalized burn rate")
    employee_morale: float = Field(..., ge=0, le=1, description="Normalized employee morale")
    product_progress: float = Field(..., ge=0, le=1, description="Normalized product progress")
    customer_satisfaction: float = Field(..., ge=0, le=1, description="Normalized customer satisfaction")
    investor_trust: float = Field(..., ge=0, le=1, description="Normalized investor trust")
    pending_tasks: float = Field(..., ge=0, le=1, description="Normalized pending tasks")
    crises: float = Field(..., ge=0, le=1, description="Normalized crises count")
    market_trend: float = Field(..., ge=0, le=1, description="Normalized market trend")
    # Extended context (4 dimensions)
    day_fraction: float = Field(..., ge=0, le=1, description="Day progress (day/max_days)")
    phase_fraction: float = Field(..., ge=0, le=1, description="Phase progress (phase_idx/2)")
    mandate_id: int = Field(..., ge=0, le=2, description="Current mandate index (0-2)")
    last_action_id: int = Field(..., ge=-1, le=12, description="Last action taken (-1 if none)")
    # Metadata
    reward: float = Field(0.0, description="Reward for the last transition")
    done: bool = Field(False, description="Episode termination flag")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")


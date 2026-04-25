from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional


ToolName = Literal[
    "finance.forecast_runway",
    "org.department_report",
    "market.risk_scan",
    "sim.what_if",
]


@dataclass(frozen=True)
class ToolDef:
    name: ToolName
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]


def _forecast_runway(args: Dict[str, Any]) -> Dict[str, Any]:
    cash = float(args.get("cash_balance", 0.0))
    burn = float(args.get("burn_rate", 0.0))
    daily_burn = max(1.0, burn)
    runway_days = cash / daily_burn
    risk = "low" if runway_days >= 90 else "medium" if runway_days >= 45 else "high"
    return {"runway_days": float(runway_days), "risk": risk}


def _department_report(args: Dict[str, Any]) -> Dict[str, Any]:
    dept = str(args.get("dept", "")).lower()
    state = args.get("state") or {}
    morale = float(state.get("employee_morale", 0.0))
    csat = float(state.get("customer_satisfaction", 0.0))
    progress = float(state.get("product_progress", 0.0))
    burn = float(state.get("burn_rate", 0.0))

    if dept in {"engineering", "engineering_manager"}:
        focus = "roadmap_delivery"
        alert = "needs_headcount" if progress < 55 else "on_track"
        suggestion = "assign_engineering_task" if progress < 80 else "launch_product"
    elif dept in {"sales", "sales_lead"}:
        focus = "pipeline_growth"
        alert = "push_revenue" if burn < 30_000 else "watch_burn"
        suggestion = "negotiate_client"
    elif dept in {"hr", "hr_recruiter"}:
        focus = "retention"
        alert = "morale_risk" if morale < 55 else "stable"
        suggestion = "improve_culture"
    elif dept in {"finance", "finance_officer"}:
        focus = "runway"
        alert = "burn_risk" if burn > 25_000 else "acceptable"
        suggestion = "reduce_costs"
    elif dept in {"cs", "customer_success"}:
        focus = "churn_risk"
        alert = "csat_risk" if csat < 60 else "stable"
        suggestion = "fix_bug_crisis"
    else:
        focus = "general"
        alert = "unknown_dept"
        suggestion = "negotiate_client"

    return {
        "dept": dept,
        "focus": focus,
        "alert": alert,
        "suggested_action": suggestion,
    }


def _risk_scan(args: Dict[str, Any]) -> Dict[str, Any]:
    state = args.get("state") or {}
    crises = float(state.get("crises", 0.0))
    csat = float(state.get("customer_satisfaction", 0.0))
    burn = float(state.get("burn_rate", 0.0))
    risk_score = 0.0
    risk_score += crises * 10.0
    risk_score += max(0.0, 60.0 - csat) * 1.2
    risk_score += max(0.0, burn - 25_000.0) / 2000.0
    risk = "low" if risk_score < 15 else "medium" if risk_score < 35 else "high"
    return {"risk": risk, "risk_score": float(risk_score)}


def _what_if(args: Dict[str, Any]) -> Dict[str, Any]:
    # Import here to keep tools module light if env deps aren’t present.
    from env.startup_env import AtlasStartupEnv

    action_idx = int(args.get("action_idx", 0))
    steps = int(args.get("steps", 3))
    preset = str(args.get("preset", "startup"))
    seed = args.get("seed", None)

    env = AtlasStartupEnv(preset=preset)
    obs, info = env.reset(seed=seed)
    total = 0.0
    for _ in range(max(1, steps)):
        _obs, reward, terminated, truncated, _info = env.step(action_idx)
        total += float(reward)
        if terminated or truncated:
            break

    return {"steps": steps, "total_reward": float(total), "final_state": env.state_snapshot()}


_TOOLS: List[ToolDef] = [
    ToolDef(
        name="finance.forecast_runway",
        description="Estimate runway in days from cash and burn.",
        input_schema={
            "type": "object",
            "properties": {
                "cash_balance": {"type": "number"},
                "burn_rate": {"type": "number"},
            },
            "required": ["cash_balance", "burn_rate"],
        },
        fn=_forecast_runway,
    ),
    ToolDef(
        name="org.department_report",
        description="Get a department report and suggested action given current state.",
        input_schema={
            "type": "object",
            "properties": {
                "dept": {"type": "string"},
                "state": {"type": "object"},
            },
            "required": ["dept", "state"],
        },
        fn=_department_report,
    ),
    ToolDef(
        name="market.risk_scan",
        description="Compute a simple market/business risk score from state.",
        input_schema={
            "type": "object",
            "properties": {"state": {"type": "object"}},
            "required": ["state"],
        },
        fn=_risk_scan,
    ),
    ToolDef(
        name="sim.what_if",
        description="Simulate a few steps ahead with a fixed action without committing.",
        input_schema={
            "type": "object",
            "properties": {
                "action_idx": {"type": "integer"},
                "steps": {"type": "integer"},
                "preset": {"type": "string"},
                "seed": {"type": ["integer", "null"]},
            },
            "required": ["action_idx"],
        },
        fn=_what_if,
    ),
]


def list_tools() -> List[Dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _TOOLS
    ]


def call_tool(name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = args or {}
    for t in _TOOLS:
        if t.name == name:
            return t.fn(args)
    raise KeyError(f"Unknown tool: {name}")


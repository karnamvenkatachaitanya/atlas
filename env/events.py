import random
from typing import Optional, Dict

EVENTS = [
    "key_employee_resigns",
    "server_outage",
    "investor_metrics_request",
    "competitor_feature_launch",
    "viral_growth",
    "lawsuit_risk",
    "sales_deal_delayed",
    "hiring_freeze",
    "market_crash",
    "customer_complaints_spike",
]

def maybe_event(state: Optional[Dict] = None, prob: float = 0.25) -> Optional[str]:
    """Trigger an event based on state conditions if available, fallback to random."""
    if not state:
        return random.choice(EVENTS) if random.random() < prob else None
        
    # We still want some stochasticity, but biased by state
    if random.random() > prob * 1.5:  # Slightly higher chance for state-based
        return None

    s = state
    morale = float(s.get("employee_morale", 100))
    progress = float(s.get("product_progress", 0))
    csat = float(s.get("customer_satisfaction", 100))
    cash = float(s.get("cash_balance", 100000))
    burn = float(s.get("burn_rate", 10000))

    candidates = []

    # State-dependent logic
    if morale < 40:
        candidates.append("key_employee_resigns")
        candidates.append("hiring_freeze")
    
    if progress > 70 and morale < 60:
        candidates.append("server_outage") # High progress, tired team
        
    if cash < burn * 2:
        candidates.append("investor_metrics_request")
        
    if progress < 30 and random.random() < 0.5:
        candidates.append("competitor_feature_launch")
        
    if csat > 85 and progress > 50:
        candidates.append("viral_growth")
        
    if csat < 50:
        candidates.append("customer_complaints_spike")
        candidates.append("lawsuit_risk")
        
    if cash > 500000 and random.random() < 0.2:
        candidates.append("market_crash")

    if random.random() < 0.1:
        candidates.append("sales_deal_delayed")

    if not candidates:
        return random.choice(EVENTS)
        
    return random.choice(candidates)

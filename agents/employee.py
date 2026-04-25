import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EmployeeAgent:
    role: str
    personality: Dict
    happiness: float = 70.0
    performance: float = 65.0
    incentives: float = 1.0
    memory: List[Dict] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)

    current_proposal: str = ""

    def propose_action(self, state: Dict) -> str:
        """Agent independently proposes an action based on its role and state."""
        proposal = ""
        if self.role == "sales_lead":
            proposal = "negotiate_client" if state.get("product_progress", 0) > 50 else "run_ads"
        elif self.role == "engineering_manager":
            proposal = "assign_engineering_task" if state.get("product_progress", 0) < 60 else "launch_product"
        elif self.role == "finance_officer":
            proposal = "reduce_costs" if state.get("burn_rate", 0) > 20000 else "raise_funding"
        elif self.role == "hr_recruiter":
            proposal = "improve_culture" if state.get("employee_morale", 0) < 50 else "hire_employee"
        elif self.role == "customer_success":
            proposal = "fix_bug_crisis" if state.get("customer_satisfaction", 0) < 60 else "improve_culture"
        else:
            proposal = "improve_culture"
            
        self.current_proposal = proposal
        return proposal

    def execute_action(self, state: Dict) -> None:
        """Agent autonomously mutates the state based on critical thresholds."""
        if self.role == "sales_lead" and state.get("revenue", 0) < 10000 and state.get("cash_balance", 0) > 50000:
            # Sales lead panics and runs ads autonomously
            state["burn_rate"] += 1000
            state["revenue"] += 1500
            self.memory.append({"action": "auto_run_ads", "message": "Autonomously ran ads to boost critical revenue."})
        elif self.role == "engineering_manager" and state.get("product_progress", 0) < 30 and state.get("employee_morale", 0) > 60:
            # Eng manager pushes team hard
            state["product_progress"] += 2
            state["employee_morale"] -= 3
            self.memory.append({"action": "auto_crunch_time", "message": "Pushed team for crunch time to get progress up."})
        elif self.role == "finance_officer" and state.get("burn_rate", 0) > 40000:
            # Finance officer autonomously cuts perks
            state["burn_rate"] -= 2000
            state["employee_morale"] -= 4
            self.memory.append({"action": "auto_cut_perks", "message": "Autonomously cut perks to reduce dangerous burn rate."})

    def negotiate(self, other_agents: List['EmployeeAgent'], state: Dict) -> str:
        """Agents negotiate with each other to align goals before proposing to CEO."""
        conflicts = 0
        for other in other_agents:
            if other.role != self.role and self.current_proposal and other.current_proposal:
                if self.current_proposal == "reduce_costs" and other.current_proposal in ["hire_employee", "run_ads", "give_bonuses"]:
                    conflicts += 1
                elif self.current_proposal == "launch_product" and other.current_proposal == "fix_bug_crisis":
                    conflicts += 1
                    
        if conflicts > 0:
            if self.role == "finance_officer":
                self.current_proposal = "raise_funding" # Compromise: instead of reducing costs, raise funding
                return f"{self.role}: Compromised to {self.current_proposal} due to department pushback on cost cutting."
            elif self.role == "engineering_manager":
                self.current_proposal = "assign_engineering_task"
                return f"{self.role}: Delayed launch to assign engineering tasks due to bug crises."
        return f"{self.role}: Holding firm on {self.current_proposal}."

    def react(self, action: str, state: Dict) -> Dict:
        """Agent reacts to the CEO's action, checking if their proposal was met."""
        msg = f"{self.role} acknowledges: {action}."
        
        # Did the CEO listen to my proposal?
        if self.current_proposal and action == self.current_proposal:
            self.happiness += 5
            self.performance += 2
            msg = f"Thanks for listening to my proposal to {action}. My team is executing."
        else:
            self.happiness -= 2
            msg = f"I proposed {self.current_proposal}, but you chose {action}. We will adapt, but morale is hurting."

        # Role-specific hard constraints
        if self.role == "sales_lead" and action in {"run_ads", "negotiate_client"}:
            self.performance += 2
        elif self.role == "engineering_manager" and action in {"assign_engineering_task", "launch_product"}:
            self.performance += 1.5
            self.happiness -= 0.5
        elif self.role == "finance_officer" and action in {"increase_salaries", "give_bonuses"}:
            self.happiness -= 5
            msg = f"WARNING: Burn rate is dangerous. Stop {action}!"
        elif self.role == "hr_recruiter" and state.get("employee_morale", 0) < 40:
            self.happiness -= 3
            msg = "URGENT: Employees are threatening to walk out."
        elif self.role == "customer_success" and state.get("customer_satisfaction", 0) < 50:
            self.performance -= 3
            msg = "URGENT: Customer churn is spiking. Fix bugs now."

        self.happiness = max(0, min(100, self.happiness + random.uniform(-1, 1)))
        self.performance = max(0, min(100, self.performance + random.uniform(-1, 1)))
        self.memory.append({"action": action, "message": msg})
        
        return {
            "message": msg,
            "happiness": self.happiness,
            "performance": self.performance,
            "role": self.role
        }

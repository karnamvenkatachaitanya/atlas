import json
import os
import random
from datasets import Dataset

# Assuming we have our heuristic policy and state formatting available
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.trl_colab_minimal import _format_prompt, _heuristic_action
from env.startup_env import ACTIONS

def process_feedback_into_dpo_dataset(feedback_file="data/human_preferences.json", output_file="data/dpo_dataset.jsonl"):
    """
    Reads the human feedback (upvotes/downvotes) from the UI and converts it into a DPO-ready dataset.
    DPO datasets require: 'prompt', 'chosen', 'rejected'
    """
    if not os.path.exists(feedback_file):
        print(f"No feedback file found at {feedback_file}")
        return

    dataset_entries = []

    with open(feedback_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line.strip())
                
                # Format the state array to match what the model expects
                s = entry.get("state", {})
                obs = [
                    s.get("cash_balance", 0),
                    s.get("revenue", 0),
                    s.get("burn_rate", 0),
                    s.get("employee_morale", 0),
                    s.get("product_progress", 0),
                    s.get("customer_satisfaction", 0),
                    s.get("investor_trust", 0),
                    s.get("pending_tasks", 0),
                    s.get("crises", 0),
                    s.get("market_trend", 0)
                ]
                
                import numpy as np
                obs_np = np.array(obs)
                
                prompt = _format_prompt(obs_np, mandate="General Management")
                
                action_taken = entry["action"]
                preference = entry["preference"]
                
                heuristic = _heuristic_action(obs_np, mandate="General Management")
                
                if preference == "upvote":
                    chosen = f"ACTION: {action_taken}"
                    # Find a rejected action (something not the chosen one)
                    rejected_action = random.choice([a for a in ACTIONS if a != action_taken])
                    rejected = f"ACTION: {rejected_action}"
                else:
                    rejected = f"ACTION: {action_taken}"
                    # If downvoted, the heuristic is likely better
                    chosen = f"ACTION: {heuristic}"
                    if chosen == rejected:
                        # Fallback if heuristic matched the bad action
                        chosen = f"ACTION: {random.choice([a for a in ACTIONS if a != action_taken])}"
                        
                dataset_entries.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
                
            except json.JSONDecodeError:
                continue
                
    if not dataset_entries:
        print("No valid entries processed.")
        return
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as out_f:
        for d in dataset_entries:
            out_f.write(json.dumps(d) + "\n")
            
    print(f"Successfully processed {len(dataset_entries)} feedback entries into DPO format at {output_file}")
    
    # Optional: also save as a Hugging Face dataset locally for immediate use
    try:
        ds = Dataset.from_list(dataset_entries)
        hf_dir = output_file.replace(".jsonl", "_hf")
        ds.save_to_disk(hf_dir)
        print(f"Saved HuggingFace dataset to {hf_dir}")
    except ImportError:
        print("Install 'datasets' to save directly as HF dataset format.")

if __name__ == "__main__":
    process_feedback_into_dpo_dataset()

import os
import sys
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.startup_env import ACTIONS, AtlasOpenEnv
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

def _format_prompt(obs, mandate) -> str:
    cash, revenue, burn_rate, morale, progress, csat, investor_trust, pending_tasks, crises, market_trend = obs.tolist()
    return (
        f"Board Mandate: {mandate}\n"
        f"Metrics: Cash:{cash:.0f}, Rev:{revenue:.0f}, Morale:{morale:.0f}, Trust:{investor_trust:.0f}\n"
        "Action: "
    )

def _parse_action_with_verifier(text: str) -> tuple[int, float]:
    """
    Format Verifier: Checks if the model output strictly contains a valid action.
    Returns: (action_idx, format_reward)
    """
    t = text.strip().lower()
    
    # Check for exact matches first (strict format compliance)
    for idx, a in enumerate(ACTIONS):
        if a.lower() == t:
            return idx, 1.0 # Reward for perfect formatting
            
    # Check for substring matches
    for idx, a in enumerate(ACTIONS):
        if a.lower() in t:
            return idx, 0.5 # Partial reward for sloppy formatting
            
    return 0, -5.0 # Heavy penalty for garbage output (Anti-Hacking/Verifier)

def main():
    print("Initializing RL Environment and PPO Trainer...")
    
    # 1. Setup PPO Config
    config = PPOConfig(
        model_name="distilgpt2",
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
    )

    # 2. Load Model, Reference Model, and Tokenizer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    env = AtlasOpenEnv(preset="startup")

    episodes = 5
    max_steps = 20 # Keep it short for demonstration

    print("\n--- Starting True RL Training Loop (with Verifiers) ---")
    for ep in range(episodes):
        obs, _ = env.reset()
        mandate = getattr(env, "mandate", "General Management")
        ep_env_reward = 0
        ep_format_reward = 0
        
        queries = []
        responses = []
        rewards = []

        for step in range(max_steps):
            # STEP 1: Give the model a prompt
            prompt_txt = _format_prompt(obs, mandate)
            query_tensor = tokenizer(prompt_txt, return_tensors="pt").input_ids[0]
            
            # STEP 2: Let it generate an action
            generation_kwargs = {
                "max_new_tokens": 5,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            response_tensor = ppo_trainer.generate(query_tensor, **generation_kwargs)
            response_only = response_tensor.squeeze()[-5:] 
            response_txt = tokenizer.decode(response_only, skip_special_tokens=True)

            # STEP 3: Multiple Independent Reward Functions / Verifiers
            action_idx, format_reward = _parse_action_with_verifier(response_txt)
            obs, env_reward, terminated, truncated, _ = env.step(action_idx)
            
            # Combine independent rewards
            total_step_reward = (float(env_reward) / 10.0) + format_reward
            
            queries.append(query_tensor)
            responses.append(response_only)
            rewards.append(torch.tensor(total_step_reward))
            
            ep_env_reward += env_reward
            ep_format_reward += format_reward

            if terminated or truncated:
                break
                
        # STEP 4: Update the model so higher-reward behavior becomes more likely
        print(f"Episode {ep+1} | Env Reward: {ep_env_reward:.2f} | Format Reward: {ep_format_reward:.2f}")
        print("Updating model weights using PPO (Backpropagation)...")
        stats = ppo_trainer.step(queries, responses, rewards)
        print(f"PPO Loss: {stats['ppo/loss/total']:.4f}\n")

    print("RL Training Complete! The model learned to maximize both Environment Metrics and Format Compliance.")

if __name__ == "__main__":
    main()

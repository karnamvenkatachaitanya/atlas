import os
import random
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from env.startup_env import ACTIONS, AtlasOpenEnv
from training.trl_colab_minimal import _format_prompt, _heuristic_action


def make_dpo_dataset(num_samples: int = 256) -> Dataset:
    """Generate preference dataset (chosen = heuristic, rejected = random)."""
    env = AtlasOpenEnv(preset="procedural")
    obs, _info = env.reset()

    prompts, chosen, rejected = [], [], []

    for _ in range(num_samples):
        mandate = getattr(env, "mandate", "General Management")
        inbox = getattr(env, "inbox", "")
        
        prompt = _format_prompt(obs, mandate, inbox)
        
        chosen_action = _heuristic_action(obs, mandate)
        
        # Pick a rejected action that is different from chosen
        rejected_action = chosen_action
        while rejected_action == chosen_action:
            rejected_action = random.choice(ACTIONS)

        prompts.append(prompt)
        chosen.append(f"ACTION: {chosen_action}")
        rejected.append(f"ACTION: {rejected_action}")

        # Step env with chosen to get realistic trajectories
        obs, _reward, terminated, truncated, _info = env.step(ACTIONS.index(chosen_action))
        if terminated or truncated:
            obs, _info = env.reset()

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected
    })


def main():
    model_name = "sshleifer/tiny-gpt2"  # Fast smoke test model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # DPO requires a reference model (copy of the base model)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model.resize_token_embeddings(len(tokenizer))

    print("Generating DPO preference dataset...")
    train_dataset = make_dpo_dataset(256)
    
    output_dir = "training/trl_dpo_out"
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing DPOTrainer...")
    # Using DPOConfig instead of TrainingArguments (TRL v0.8.0+)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=0.1,  # KL penalty
        per_device_train_batch_size=4,
        max_steps=50,
        learning_rate=1e-4,
        remove_unused_columns=False,
        report_to="none",
        max_prompt_length=1024,
        max_length=1200,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("Training DPO...")
    trainer.train()

    print(f"Saving DPO aligned model to {output_dir}")
    trainer.save_model(output_dir)
    print("DPO stage complete!")


if __name__ == "__main__":
    main()

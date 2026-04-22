import os
import sys
from typing import List, Tuple

import numpy as np

# Ensure project root is importable when running "python training/trl_colab_minimal.py".
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import an OpenEnv-core symbol so the notebook demonstrates "usage of OpenEnv (latest release)".
# This does not change the environment logic; it just ensures the OpenEnv package is present.
from openenv.core import GenericEnvClient  # noqa: F401,E402

from env.startup_env import ACTIONS, AtlasOpenEnv  # noqa: E402


def _format_prompt(obs: np.ndarray) -> str:
    # obs is shape (10,) from AtlasStartupEnv._obs()
    (
        cash,
        revenue,
        burn_rate,
        morale,
        progress,
        csat,
        investor_trust,
        pending_tasks,
        crises,
        market_trend,
    ) = obs.tolist()

    return (
        "You are the CEO agent in a startup simulation.\n"
        "Choose ONE action from the action list that improves long-term reward.\n\n"
        f"State:\n"
        f"- cash_balance: {cash:.0f}\n"
        f"- revenue: {revenue:.0f}\n"
        f"- burn_rate: {burn_rate:.0f}\n"
        f"- employee_morale: {morale:.1f}\n"
        f"- product_progress: {progress:.1f}\n"
        f"- customer_satisfaction: {csat:.1f}\n"
        f"- investor_trust: {investor_trust:.1f}\n"
        f"- pending_tasks: {pending_tasks:.1f}\n"
        f"- crises: {crises:.1f}\n"
        f"- market_trend: {market_trend:.1f}\n\n"
        "Actions:\n"
        + "\n".join([f"- {a}" for a in ACTIONS])
        + "\n\nAnswer with exactly one action name.\n"
        "Action: "
    )


def _heuristic_action(obs: np.ndarray) -> str:
    # Minimal "teacher" policy: easy to justify, deterministic, and environment-driven.
    cash, revenue, burn_rate, morale, progress, csat, investor_trust, pending_tasks, crises, market_trend = (
        obs.tolist()
    )
    if cash < 100_000:
        return "reduce_costs"
    if csat < 60:
        return "fix_bug_crisis"
    if progress < 55:
        return "assign_engineering_task"
    return "launch_product"


def make_dataset(num_samples: int = 128) -> List[Tuple[str, str]]:
    env = AtlasOpenEnv(preset="startup")
    obs, _info = env.reset()

    pairs: List[Tuple[str, str]] = []
    for _ in range(num_samples):
        prompt = _format_prompt(obs)
        action_name = _heuristic_action(obs)
        pairs.append((prompt, action_name))

        action_idx = ACTIONS.index(action_name)
        obs, _reward, terminated, truncated, _info = env.step(action_idx)
        if terminated or truncated:
            obs, _info = env.reset()

    return pairs


def main() -> None:
    """
    Colab-friendly minimal TRL training example.

    This *uses the environment* to generate (state -> action) pairs, then uses TRL's
    SFTTrainer to fine-tune a tiny language model to imitate the heuristic policy.
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    pairs = make_dataset(num_samples=128)
    ds = Dataset.from_dict(
        {
            "text": [p + a for (p, a) in pairs],
        }
    )

    model_name = os.environ.get("ATLAS_TRL_MODEL", "distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    out_dir = os.path.join("training", "trl_out")
    cfg = SFTConfig(
        output_dir=out_dir,
        max_steps=30,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=5,
        save_steps=30,
        dataset_text_field="text",
        # Colab often runs this notebook on CPU. Force CPU-safe settings so
        # Transformers doesn't try bf16/fp16 GPU paths.
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        packing=False,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Saved TRL SFT model to: {out_dir}")


if __name__ == "__main__":
    main()


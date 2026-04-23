import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

try:  # Optional but imported early for optimization compatibility.
    import unsloth  # noqa: F401
except Exception:  # pragma: no cover
    unsloth = None

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# Ensure project root is importable when running "python training/trl_ppo_rl.py".
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.startup_env import ACTIONS, AtlasOpenEnv  # noqa: E402

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLMWithValueHead = None  # type: ignore[assignment]
    PPOConfig = None  # type: ignore[assignment]
    PPOTrainer = None  # type: ignore[assignment]


HAS_LEGACY_PPO_API = bool(
    PPOTrainer is not None and hasattr(PPOTrainer, "step") and hasattr(PPOTrainer, "generate")
)


ACTION_TOKENS = [f"<a{i}>" for i in range(len(ACTIONS))]


@dataclass
class RunConfig:
    model_name: str = os.environ.get("ATLAS_RL_MODEL", "sshleifer/tiny-gpt2")
    episodes: int = int(os.environ.get("ATLAS_RL_EPISODES", "16"))
    max_steps_per_episode: int = int(os.environ.get("ATLAS_RL_MAX_STEPS", "180"))
    learning_rate: float = float(os.environ.get("ATLAS_RL_LR", "1.0e-5"))
    gamma: float = float(os.environ.get("ATLAS_RL_GAMMA", "0.99"))
    output_dir: str = os.environ.get("ATLAS_RL_OUT", os.path.join("training", "trl_ppo_out"))
    use_curriculum: bool = os.environ.get("ATLAS_RL_CURRICULUM", "1") == "1"


CURRICULUM_STAGES = [
    {
        "name": "easy",
        "preset": "growth",
        "max_steps": 60,
        "advance_reward": 1000.0,
        "min_episodes": 3,
    },
    {
        "name": "medium",
        "preset": "startup",
        "max_steps": 120,
        "advance_reward": 2200.0,
        "min_episodes": 3,
    },
    {
        "name": "hard",
        "preset": "crisis",
        "max_steps": 180,
        "advance_reward": 2600.0,
        "min_episodes": 0,
    },
]


def _format_prompt(obs: np.ndarray, mandate: str) -> str:
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
        "You are an AI CEO in a startup simulation.\n"
        f"Board Mandate: {mandate}\n\n"
        "Choose exactly one action token from the allowed list.\n"
        "Valid tokens:\n"
        + "\n".join([f"- {token} = {name}" for token, name in zip(ACTION_TOKENS, ACTIONS)])
        + "\n\n"
        "Current state:\n"
        f"cash_balance={cash:.0f}, revenue={revenue:.0f}, burn_rate={burn_rate:.0f}, "
        f"employee_morale={morale:.1f}, product_progress={progress:.1f}, "
        f"customer_satisfaction={csat:.1f}, investor_trust={investor_trust:.1f}, "
        f"pending_tasks={pending_tasks:.1f}, crises={crises:.1f}, market_trend={market_trend:.1f}\n\n"
        "Action token:"
    )


def _build_token_maps(tokenizer: AutoTokenizer) -> Tuple[List[int], Dict[int, int]]:
    added = tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
    if added:
        print(f"Added {added} action-special tokens.")
    token_ids = tokenizer.convert_tokens_to_ids(ACTION_TOKENS)
    id_to_action = {tid: idx for idx, tid in enumerate(token_ids)}
    return token_ids, id_to_action


def _decode_action(response_tensor: torch.Tensor, id_to_action: Dict[int, int]) -> int | None:
    if response_tensor.numel() == 0:
        return None
    first_token = int(response_tensor[0].item())
    return id_to_action.get(first_token)


def _discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    out: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        out.append(running)
    out.reverse()
    return out


def _load_policy_model(model_name: str):
    # Use the TRL value-head model so PPO can optimize policy + value estimates.
    if AutoModelForCausalLMWithValueHead is None:
        raise RuntimeError("TRL legacy PPO API unavailable; use REINFORCE fallback path.")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _load_reinforce_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _run_reinforce_curriculum(cfg: RunConfig) -> None:
    model, tokenizer = _load_reinforce_model(cfg.model_name)
    action_token_ids, id_to_action = _build_token_maps(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    env = AtlasOpenEnv(preset="startup")
    episode_rewards: List[float] = []
    stage_idx = 0
    stage_episode_count = 0
    recent_stage_rewards: List[float] = []

    for ep in range(cfg.episodes):
        if cfg.use_curriculum:
            stage = CURRICULUM_STAGES[stage_idx]
            env.core.preset = stage["preset"]
            episode_step_cap = min(cfg.max_steps_per_episode, int(stage["max_steps"]))
            print(
                f"Curriculum stage={stage['name']} preset={stage['preset']} "
                f"max_steps={episode_step_cap}"
            )
        else:
            stage = {
                "name": "no_curriculum",
                "preset": "startup",
                "max_steps": cfg.max_steps_per_episode,
                "advance_reward": float("inf"),
                "min_episodes": 0,
            }
            env.core.preset = stage["preset"]
            episode_step_cap = cfg.max_steps_per_episode

        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        step_rewards: List[float] = []
        log_probs: List[torch.Tensor] = []

        while not done and steps < episode_step_cap:
            mandate = getattr(env, "mandate", "Balanced Stability")
            prompt = _format_prompt(obs, mandate)
            inputs = tokenizer(prompt, return_tensors="pt")
            logits = model(**inputs).logits[0, -1, action_token_ids]
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            sampled_idx = dist.sample()
            sampled_token_id = int(action_token_ids[int(sampled_idx.item())])
            action_idx = id_to_action.get(sampled_token_id)

            if action_idx is None:
                action_idx = random.randint(0, len(ACTIONS) - 1)
                reward_penalty = -2.0
            else:
                reward_penalty = 0.0

            obs, reward, terminated, truncated, _info = env.step(action_idx)
            reward = float(reward) + reward_penalty
            done = bool(terminated or truncated)

            log_probs.append(dist.log_prob(sampled_idx))
            step_rewards.append(reward)
            total_reward += reward
            steps += 1

        returns = _discounted_returns(step_rewards, cfg.gamma)
        if returns and log_probs:
            returns_t = torch.tensor(returns, dtype=torch.float32)
            if returns_t.std().item() > 1e-6:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
            loss = -torch.stack([lp * ret for lp, ret in zip(log_probs, returns_t)]).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        episode_rewards.append(total_reward)
        stage_episode_count += 1
        recent_stage_rewards.append(total_reward)
        if len(recent_stage_rewards) > 3:
            recent_stage_rewards = recent_stage_rewards[-3:]

        if cfg.use_curriculum and stage_idx < len(CURRICULUM_STAGES) - 1:
            ready_by_episode_count = stage_episode_count >= int(stage["min_episodes"])
            rolling = float(np.mean(recent_stage_rewards)) if recent_stage_rewards else 0.0
            ready_by_reward = rolling >= float(stage["advance_reward"])
            if ready_by_episode_count and ready_by_reward:
                stage_idx += 1
                stage_episode_count = 0
                recent_stage_rewards = []
                next_stage = CURRICULUM_STAGES[stage_idx]
                print(
                    f"Promoted to stage={next_stage['name']} preset={next_stage['preset']} "
                    f"after rolling_reward={rolling:.2f}"
                )

        print(
            f"Episode {ep + 1:02d}/{cfg.episodes} | stage={stage['name']} "
            f"steps={steps} | total_reward={total_reward:.2f}"
        )

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    rewards_np = np.array(episode_rewards, dtype=np.float32)
    print(f"Saved RL policy to: {cfg.output_dir}")
    print(f"Mean episode reward: {float(rewards_np.mean()):.2f}")
    print(f"Best episode reward: {float(rewards_np.max()):.2f}")


def main() -> None:
    cfg = RunConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # If Unsloth is available we report it explicitly; PPO still uses TRL's value-head model.
    if unsloth is not None:
        print("Unsloth detected in environment: fast kernels available for SFT phase.")
    else:
        print("Unsloth not detected at runtime; continuing with TRL PPO training.")

    if not HAS_LEGACY_PPO_API:
        print("Detected TRL PPO API variant without step/generate. Using REINFORCE fallback.")
        _run_reinforce_curriculum(cfg)
        return

    model, tokenizer = _load_policy_model(cfg.model_name)
    action_token_ids, id_to_action = _build_token_maps(tokenizer)

    # Keep embeddings aligned when new action tokens are added.
    model.pretrained_model.resize_token_embeddings(len(tokenizer))

    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=False,
        fp16=False,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
    )

    env = AtlasOpenEnv(preset="startup")
    episode_rewards: List[float] = []
    stage_idx = 0
    stage_episode_count = 0
    recent_stage_rewards: List[float] = []

    for ep in range(cfg.episodes):
        if cfg.use_curriculum:
            stage = CURRICULUM_STAGES[stage_idx]
            env.core.preset = stage["preset"]
            episode_step_cap = min(cfg.max_steps_per_episode, int(stage["max_steps"]))
            print(
                f"Curriculum stage={stage['name']} preset={stage['preset']} "
                f"max_steps={episode_step_cap}"
            )
        else:
            stage = {
                "name": "no_curriculum",
                "preset": "startup",
                "max_steps": cfg.max_steps_per_episode,
                "advance_reward": float("inf"),
                "min_episodes": 0,
            }
            env.core.preset = stage["preset"]
            episode_step_cap = cfg.max_steps_per_episode

        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        step_rewards: List[float] = []
        step_data: List[Tuple[torch.Tensor, torch.Tensor]] = []

        while not done and steps < episode_step_cap:
            mandate = getattr(env, "mandate", "Balanced Stability")
            prompt = _format_prompt(obs, mandate)
            query_tensor = tokenizer.encode(prompt, return_tensors="pt")[0]

            response_tensor = ppo_trainer.generate(
                [query_tensor],
                return_prompt=False,
                max_new_tokens=1,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

            action_idx = _decode_action(response_tensor, id_to_action)
            if action_idx is None:
                # Penalize invalid actions so the policy learns to emit valid action tokens.
                action_idx = random.randint(0, len(ACTIONS) - 1)
                reward_penalty = -2.0
            else:
                reward_penalty = 0.0

            obs, reward, terminated, truncated, _info = env.step(action_idx)
            reward = float(reward) + reward_penalty
            done = bool(terminated or truncated)

            step_data.append((query_tensor, response_tensor))
            step_rewards.append(reward)
            total_reward += reward
            steps += 1

        # Use discounted returns so long-horizon credit assignment is represented in PPO updates.
        returns = _discounted_returns(step_rewards, cfg.gamma)
        for (query_tensor, response_tensor), ret in zip(step_data, returns):
            ppo_trainer.step(
                [query_tensor],
                [response_tensor],
                [torch.tensor(ret, dtype=torch.float32)],
            )

        episode_rewards.append(total_reward)
        stage_episode_count += 1
        recent_stage_rewards.append(total_reward)
        if len(recent_stage_rewards) > 3:
            recent_stage_rewards = recent_stage_rewards[-3:]

        # Curriculum promotion rule:
        # Move to harder stage only after the agent demonstrates consistent non-zero progress.
        if cfg.use_curriculum and stage_idx < len(CURRICULUM_STAGES) - 1:
            ready_by_episode_count = stage_episode_count >= int(stage["min_episodes"])
            rolling = float(np.mean(recent_stage_rewards)) if recent_stage_rewards else 0.0
            ready_by_reward = rolling >= float(stage["advance_reward"])
            if ready_by_episode_count and ready_by_reward:
                stage_idx += 1
                stage_episode_count = 0
                recent_stage_rewards = []
                next_stage = CURRICULUM_STAGES[stage_idx]
                print(
                    f"Promoted to stage={next_stage['name']} preset={next_stage['preset']} "
                    f"after rolling_reward={rolling:.2f}"
                )

        print(
            f"Episode {ep + 1:02d}/{cfg.episodes} | stage={stage['name']} "
            f"steps={steps} | total_reward={total_reward:.2f}"
        )

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    rewards_np = np.array(episode_rewards, dtype=np.float32)
    print(f"Saved PPO RL policy to: {cfg.output_dir}")
    print(f"Mean episode reward: {float(rewards_np.mean()):.2f}")
    print(f"Best episode reward: {float(rewards_np.max()):.2f}")


if __name__ == "__main__":
    main()

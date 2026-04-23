import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.startup_env import AtlasStartupEnv


@dataclass
class ValidationResult:
    condition_1_stepwise_actions: bool
    condition_2_code_verifiable_success: bool
    condition_3_challenging_but_possible: bool
    details: Dict[str, float]


def random_policy(env: AtlasStartupEnv) -> int:
    return int(env.action_space.sample())


def heuristic_policy(env: AtlasStartupEnv) -> int:
    s = env.state
    if s["cash_balance"] < 100000:
        return 7  # reduce_costs
    if s["customer_satisfaction"] < 60:
        return 9  # fix_bug_crisis
    if s["product_progress"] < 55:
        return 3  # assign_engineering_task
    return 4  # launch_product


def run_episode(env: AtlasStartupEnv, policy_fn: Callable[[AtlasStartupEnv], int]) -> Tuple[float, int]:
    env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = policy_fn(env)
        _, reward, terminated, truncated, _ = env.step(action)
        if not isinstance(reward, (float, int, np.floating)):
            raise TypeError(f"Reward must be numeric, got {type(reward)}")
        total_reward += float(reward)
        done = bool(terminated or truncated)
        steps += 1

    return float(total_reward), int(steps)


def validate(num_eval_episodes: int = 120) -> ValidationResult:
    # Condition (1): model can act step-by-step (multi-step action loop exists and is usable).
    env = AtlasStartupEnv(preset="startup")
    _, steps = run_episode(env, random_policy)
    condition_1 = steps > 1 and int(env.action_space.n) > 1

    # Condition (2): success can be checked by code.
    # We verify that the environment emits numeric rewards and cumulative reward can be computed.
    sample_rewards: List[float] = []
    sample_breakdowns: List[dict] = []
    for _ in range(5):
        env2 = AtlasStartupEnv(preset="startup")
        reward, _ = run_episode(env2, random_policy)
        sample_rewards.append(reward)
        env3 = AtlasStartupEnv(preset="startup")
        env3.reset()
        _, _, _, _, info = env3.step(random_policy(env3))
        sample_breakdowns.append(info.get("reward_breakdown", {}))
    condition_2 = all(np.isfinite(sample_rewards)) and all("business_reward" in bd for bd in sample_breakdowns)

    # Condition (3): challenging but possible.
    # We set success threshold from the random policy's 90th percentile so random has ~10% success.
    random_rewards = np.array(
        [run_episode(AtlasStartupEnv(preset="startup"), random_policy)[0] for _ in range(num_eval_episodes)],
        dtype=np.float64,
    )
    threshold = float(np.quantile(random_rewards, 0.90))
    random_success = float(np.mean(random_rewards >= threshold))

    heuristic_rewards = np.array(
        [run_episode(AtlasStartupEnv(preset="startup"), heuristic_policy)[0] for _ in range(num_eval_episodes)],
        dtype=np.float64,
    )
    heuristic_success = float(np.mean(heuristic_rewards >= threshold))

    # Pass criteria:
    # - random has non-zero but not dominant success probability (about 5%-20%),
    # - stronger policy shows clear learnable headroom.
    condition_3 = (0.05 <= random_success <= 0.20) and (heuristic_success >= random_success + 0.20)

    return ValidationResult(
        condition_1_stepwise_actions=condition_1,
        condition_2_code_verifiable_success=condition_2,
        condition_3_challenging_but_possible=condition_3,
        details={
            "random_reward_mean": float(np.mean(random_rewards)),
            "heuristic_reward_mean": float(np.mean(heuristic_rewards)),
            "success_threshold": threshold,
            "random_success_rate": random_success,
            "heuristic_success_rate": heuristic_success,
            "steps_per_episode": float(steps),
        },
    )


def main() -> None:
    result = validate(num_eval_episodes=120)

    print("Condition (1) Step-by-step actions:", "PASS" if result.condition_1_stepwise_actions else "FAIL")
    print(
        "Condition (2) Code-verifiable success:",
        "PASS" if result.condition_2_code_verifiable_success else "FAIL",
    )
    print(
        "Condition (3) Challenging but possible:",
        "PASS" if result.condition_3_challenging_but_possible else "FAIL",
    )
    print("Details:")
    for k, v in result.details.items():
        print(f"- {k}: {v:.4f}")

    all_pass = (
        result.condition_1_stepwise_actions
        and result.condition_2_code_verifiable_success
        and result.condition_3_challenging_but_possible
    )
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

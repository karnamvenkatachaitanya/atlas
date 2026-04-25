import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is importable when running "python training/train.py".
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.startup_env import AtlasStartupEnv


def run_episode(env: AtlasStartupEnv, policy: str = "random") -> float:
    obs, info = env.reset()  # Get normalized obs
    done = False
    total_reward = 0.0
    while not done:
        if policy == "random":
            action = env.action_space.sample()
        else:
            # Use state from info dict (unnormalized) for heuristic decisions
            state = env.state_snapshot()
            if state["cash_balance"] < 100000:
                action = 7  # reduce_costs
            elif state["customer_satisfaction"] < 60:
                action = 9  # fix_bug_crisis
            elif state["product_progress"] < 55:
                action = 3  # assign_engineering_task
            else:
                action = 4  # launch_product
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


def main() -> None:
    env = AtlasStartupEnv(preset="startup")
    random_rewards = [run_episode(env, "random") for _ in range(20)]
    trained_rewards = [run_episode(env, "heuristic") for _ in range(20)]

    print(f"Random avg reward: {np.mean(random_rewards):.2f}")
    print(f"Trained-like avg reward: {np.mean(trained_rewards):.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(random_rewards, label="Random")
    plt.plot(trained_rewards, label="Trained-like")
    plt.title("ATLAS Reward Curve: Before vs After")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training/reward_curve.png")
    print("Saved reward curve to training/reward_curve.png")


if __name__ == "__main__":
    main()

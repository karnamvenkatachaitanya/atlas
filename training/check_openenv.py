import os
import sys

# Ensure project root is importable when running "python training/check_openenv.py".
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.startup_env import AtlasOpenEnv


def main() -> None:
    env = AtlasOpenEnv(preset="startup")
    obs, info = env.reset()
    print("OpenEnv adapter reset ok")
    print("obs_shape:", getattr(obs, "shape", None))
    print("info:", info)

    for step_idx in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step_idx + 1} action={info.get('action_name')} reward={reward:.2f} "
            f"done={terminated or truncated}"
        )

    print("render:", env.render())
    print("OpenEnv adapter check passed.")


if __name__ == "__main__":
    main()

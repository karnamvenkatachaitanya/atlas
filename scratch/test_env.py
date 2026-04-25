from env.startup_env import AtlasOpenEnv
try:
    env = AtlasOpenEnv(preset="startup")
    obs, info = env.reset()
    print("Reset successful")
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step successful, reward: {reward}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

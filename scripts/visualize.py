import os
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "storage" / "ppo_pickcube.zip"

os.environ.setdefault("SAPIEN_RENDER_MODE", "cpu")

model = PPO.load(MODEL_PATH, device="cpu")
env = gym.make("PickCube-v1", obs_mode="state", render_mode="human")
obs, _ = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

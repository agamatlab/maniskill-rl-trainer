import mani_skill.envs
import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_pickcube", device="cpu")
env = gym.make("PickCube-v1", obs_mode="state", render_mode="human")    
obs, _ = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
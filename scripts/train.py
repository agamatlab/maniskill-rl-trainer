import mani_skill.envs
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

def make_env():
    env = gym.make("PickCube-v1", num_envs=1, obs_mode="state")
    env = FlattenObservation(env)  # PPO expects flat Box
    return env

env = make_env()
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, gamma=0.99,
            gae_lambda=0.95, ent_coef=0.0, learning_rate=3e-4, clip_range=0.2,
            verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_pickcube")
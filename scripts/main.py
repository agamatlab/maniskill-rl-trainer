import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mani_skill.envs  # registers environments
import gymnasium as gym
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

device = torch.device("cpu")
env = CPUGymWrapper(gym.make("PickCube-v1", num_envs=1))
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
gamma = 0.99

for update in range(10000):
    obs, _ = env.reset()
    log_probs = []
    rewards = []
    done = False
    
    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        dist = policy(obs_tensor)
        action = dist.sample()
        log_probs.append(dist.log_prob(action).sum(dim=-1))

        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
        terminated = bool(np.array(terminated).item())
        truncated = bool(np.array(truncated).item())
        done = terminated or truncated

    returns = []
    G = torch.zeros(1, device=device)

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G.detach())

    returns = torch.stack(returns)
    log_probs = torch.stack(log_probs)
    loss = - (log_probs * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if update % 10 == 0:
        print(f"Update {update}, Return: {returns[0].item():.2f}")

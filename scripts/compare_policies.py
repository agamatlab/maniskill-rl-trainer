from __future__ import annotations

import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import random
import numpy as np
import torch
import tyro
import torch.nn as nn
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def _load_module(module_name: str, relative_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module at {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_device(data, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    return data


def _stats_from_tensor(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "num_samples": 0.0}
    mean = values.mean().item()
    std = values.std(unbiased=False).item() if values.numel() > 1 else 0.0
    return {"mean": mean, "std": std, "num_samples": float(values.numel())}


def _stats_from_list(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "num_samples": 0.0}
    tensor = torch.tensor(vals, dtype=torch.float32)
    return _stats_from_tensor(tensor)


def _maybe_adapt_my_ppo_agent(agent, state_dict):
    legacy_key = "feature_net.stateEncoder.extractors.state.weight"
    if legacy_key not in state_dict:
        return
    state_encoder = agent.feature_net.stateEncoder
    if "state" not in state_encoder.extractors:
        return
    extractor = state_encoder.extractors["state"]
    if isinstance(extractor, nn.Sequential):
        first_linear = None
        for layer in extractor:
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break
        if first_linear is None:
            raise RuntimeError("Unable to adapt my_ppo state encoder for legacy checkpoint.")
        legacy_linear = nn.Linear(first_linear.in_features, first_linear.out_features)
        state_encoder.extractors["state"] = legacy_linear
        state_encoder.out_features = first_linear.out_features
    agent.feature_net.fusion_network = nn.Identity()
    agent.feature_net.out_features = (
        agent.feature_net.stateEncoder.out_features + agent.feature_net.visualEncoder.out_features
    )


@dataclass
class CompareArgs:
    env_id: str = "PickCube-v1"
    seed: int = 1
    cuda: bool = False
    num_eval_envs: int = 1
    num_eval_steps: int = 500
    target_episodes: Optional[int] = 10
    eval_reconfiguration_freq: Optional[int] = 1
    eval_partial_reset: bool = False
    control_mode: Optional[str] = "pd_joint_delta_pos"
    sim_backend: str = "cpu"
    render_mode: str = "none"
    include_state: bool = True
    deterministic: bool = True
    ppo_rgbd_checkpoint: Optional[str] = None
    my_ppo_checkpoint: Optional[str] = None


def make_eval_env(args: CompareArgs):
    env_kwargs = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend=args.sim_backend)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=args.include_state)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(
        env,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )


def evaluate_agent(
    label: str,
    module,
    checkpoint_path: Optional[str],
    args: CompareArgs,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    eval_env = make_eval_env(args)
    try:
        obs, _ = eval_env.reset(seed=args.seed)
    except RuntimeError as exc:
        eval_env.close()
        raise RuntimeError(f"{label}: failed to reset environment ({exc})") from exc
    obs = _to_device(obs, device)

    agent = module.Agent(eval_env, sample_obs=obs)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if label == "my_ppo":
            _maybe_adapt_my_ppo_agent(agent, state_dict)
        agent.load_state_dict(state_dict)
    agent.to(device)
    agent.eval()

    episode_returns = torch.zeros(args.num_eval_envs, device=device)
    episode_lengths = torch.zeros(args.num_eval_envs, device=device)
    collected_returns = []
    collected_lengths = []
    eval_metrics = defaultdict(list)

    steps = 0
    while steps < args.num_eval_steps and (
        args.target_episodes is None or len(collected_returns) < args.target_episodes
    ):
        with torch.no_grad():
            actions = agent.get_action(obs, deterministic=args.deterministic)
        next_obs, reward, terminations, truncations, infos = eval_env.step(actions)
        reward = reward.to(device)
        done = torch.logical_or(terminations.to(device), truncations.to(device))

        episode_returns += reward
        episode_lengths += 1

        if "final_info" in infos:
            mask = infos["_final_info"]
            if mask.any():
                for key, value in infos["final_info"]["episode"].items():
                    eval_metrics[key].append(value[mask].detach().cpu())

        for env_idx, finished in enumerate(done):
            if finished:
                collected_returns.append(episode_returns[env_idx].item())
                collected_lengths.append(episode_lengths[env_idx].item())
                episode_returns[env_idx] = 0
                episode_lengths[env_idx] = 0

        obs = _to_device(next_obs, device)
        steps += 1

    eval_env.close()

    summary: Dict[str, Dict[str, float]] = {}
    summary["episode_return"] = _stats_from_list(collected_returns)
    summary["episode_length"] = _stats_from_list(collected_lengths)
    summary["episodes_sampled"] = {"mean": float(len(collected_returns)), "std": 0.0, "num_samples": 1.0}

    for metric_name, tensors in eval_metrics.items():
        stacked = torch.cat([t.reshape(-1).float() for t in tensors]) if tensors else torch.tensor([])
        summary[f"{metric_name}"] = _stats_from_tensor(stacked)

    if args.target_episodes is not None and len(collected_returns) < args.target_episodes:
        deficit = args.target_episodes - len(collected_returns)
        summary["warning"] = {
            "mean": float(deficit),
            "std": 0.0,
            "num_samples": 1.0,
        }

    return summary


def format_summary(label: str, metrics: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{label}"]
    for key, stats in sorted(metrics.items()):
        mean = stats["mean"]
        std = stats["std"]
        count = stats["num_samples"]
        lines.append(f"  {key:20s}: mean={mean:.4f} std={std:.4f} n={count:.0f}")
    return "\n".join(lines)


def main(args: CompareArgs):
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    script_dir = Path(__file__).resolve().parent
    ppo_rgbd_module = _load_module("ppo_rgbd_module", script_dir / "ppo_rgbd.py")
    my_ppo_module = _load_module("my_ppo_module", script_dir / "my_ppo.py")

    comparisons = [
        ("ppo_rgbd", ppo_rgbd_module, args.ppo_rgbd_checkpoint),
        ("my_ppo", my_ppo_module, args.my_ppo_checkpoint),
    ]

    for label, module, ckpt in comparisons:
        if ckpt is None:
            print(f"Skipping {label}: checkpoint path not provided.")
            continue
        try:
            metrics = evaluate_agent(label, module, ckpt, args, device)
        except RuntimeError as exc:
            print(f"{label} evaluation failed: {exc}")
            continue
        print(format_summary(label, metrics))
        print("-" * 60)


if __name__ == "__main__":
    main(tyro.cli(CompareArgs))

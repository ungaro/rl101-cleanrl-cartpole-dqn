"""Evaluate a trained DQN model with rendering.

Loads a saved model and runs it on the environment so you can
watch the trained agent in action.

Run: python scripts/evaluate.py --model-path runs/.../model.pth
     python scripts/evaluate.py --model-path runs/.../model.pth --env-id LunarLander-v3
"""

import argparse
import sys

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """Same architecture as CleanRL's DQN Q-network."""

    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model (.pth)")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make(args.env_id, render_mode="human")

    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    q_network.eval()

    print(f"Loaded model from {args.model_path}")
    print(f"Environment: {args.env_id}")
    print(f"Running {args.episodes} episodes...\n")

    rewards = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0
        step = 0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(obs_tensor)
            action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"Episode {ep + 1}: {step} steps, reward: {total_reward}")

    print(f"\nMean reward: {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})")
    env.close()


if __name__ == "__main__":
    main()

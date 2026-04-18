"""Train PPO on Atari games using CleanRL's ppo_atari.py.

Week 3 of the RL 101 study group — PPO beyond CartPole.
Calls CleanRL's Atari PPO implementation with tuned defaults.

Supported games:
  python scripts/train_atari_ppo.py                     # Breakout (default)
  python scripts/train_atari_ppo.py --game pong         # Pong
  python scripts/train_atari_ppo.py --game spaceinvaders # Space Invaders
  python scripts/train_atari_ppo.py --game breakout     # Breakout

Pass --short for a quick 1M-step demo run instead of the full 10M.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PPO_ATARI_SCRIPT = Path(__file__).resolve().parent.parent / "cleanrl" / "cleanrl" / "ppo_atari.py"

GAMES = {
    "breakout": {
        "env_id": "BreakoutNoFrameskip-v4",
        "total_timesteps": "10000000",
        "description": "Break bricks with a bouncing ball. Emergent tunneling strategy.",
    },
    "pong": {
        "env_id": "PongNoFrameskip-v4",
        "total_timesteps": "5000000",
        "description": "Beat the CPU at Pong. One of the easiest Atari games for RL.",
    },
    "spaceinvaders": {
        "env_id": "SpaceInvadersNoFrameskip-v4",
        "total_timesteps": "10000000",
        "description": "Shoot aliens while dodging return fire. Offense/defense balance.",
    },
}

SHORT_TIMESTEPS = "1000000"  # 1M steps for quick demo


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Atari games")
    parser.add_argument(
        "--game",
        choices=list(GAMES.keys()),
        default="breakout",
        help="Which Atari game to train on (default: breakout)",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Quick 1M-step demo run instead of full training",
    )
    args, extra = parser.parse_known_args()

    if not PPO_ATARI_SCRIPT.exists():
        print(f"Error: CleanRL Atari PPO not found at {PPO_ATARI_SCRIPT}")
        print("Run 'make setup' or 'bash setup.sh' first.")
        sys.exit(1)

    game = GAMES[args.game]
    timesteps = SHORT_TIMESTEPS if args.short else game["total_timesteps"]

    cmd = [
        sys.executable, str(PPO_ATARI_SCRIPT),
        "--env-id", game["env_id"],
        "--total-timesteps", timesteps,
        "--learning-rate", "2.5e-4",
        "--num-envs", "8",
        "--num-steps", "128",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--num-minibatches", "4",
        "--update-epochs", "4",
        "--clip-coef", "0.1",
        "--ent-coef", "0.01",
        "--vf-coef", "0.5",
        "--capture-video",
    ]

    # Pass through any extra CleanRL args
    cmd.extend(extra)

    print("=" * 60)
    print(f"Training PPO on {game['env_id']}")
    print(f"  {game['description']}")
    print(f"  Timesteps: {timesteps}")
    print("=" * 60)
    print(f"Script: {PPO_ATARI_SCRIPT}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    # ppo_atari.py imports from cleanrl_utils which lives under cleanrl/
    # Set PYTHONPATH so it can find cleanrl_utils, but run from project root
    # so videos/ and runs/ land in the project directory.
    cleanrl_root = PPO_ATARI_SCRIPT.parent.parent
    project_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cleanrl_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=str(project_root), env=env)


if __name__ == "__main__":
    main()

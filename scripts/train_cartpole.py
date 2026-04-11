"""Train DQN on CartPole-v1 using CleanRL's dqn.py.

Calls CleanRL's implementation with tuned defaults for CartPole.
Run: python scripts/train_cartpole.py
"""

import subprocess
import sys
from pathlib import Path

# CleanRL's dqn.py location
DQN_SCRIPT = Path(__file__).resolve().parent.parent / "cleanrl" / "cleanrl" / "dqn.py"


def main():
    if not DQN_SCRIPT.exists():
        print(f"Error: CleanRL not found at {DQN_SCRIPT}")
        print("Run 'make setup' or 'bash setup.sh' first.")
        sys.exit(1)

    cmd = [
        sys.executable, str(DQN_SCRIPT),
        "--env-id", "CartPole-v1",
        "--total-timesteps", "500000",
        "--learning-rate", "2.5e-4",
        "--buffer-size", "10000",
        "--batch-size", "128",
        "--gamma", "0.99",
        "--target-network-frequency", "500",
        "--start-e", "1.0",
        "--end-e", "0.05",
        "--exploration-fraction", "0.1",
        "--train-frequency", "4",
        "--save-model",
        "--capture-video",
    ]

    # Pass through any extra args
    cmd.extend(sys.argv[1:])

    print("=" * 60)
    print("Training DQN on CartPole-v1")
    print("=" * 60)
    print(f"Script: {DQN_SCRIPT}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

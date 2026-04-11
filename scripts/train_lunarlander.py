"""Train DQN on LunarLander-v3 using CleanRL's dqn.py.

Bonus demo — harder environment, takes longer to train.
Run: python scripts/train_lunarlander.py
"""

import subprocess
import sys
from pathlib import Path

DQN_SCRIPT = Path(__file__).resolve().parent.parent / "cleanrl" / "cleanrl" / "dqn.py"


def main():
    if not DQN_SCRIPT.exists():
        print(f"Error: CleanRL not found at {DQN_SCRIPT}")
        print("Run 'make setup' or 'bash setup.sh' first.")
        sys.exit(1)

    cmd = [
        sys.executable, str(DQN_SCRIPT),
        "--env-id", "LunarLander-v3",
        "--total-timesteps", "1000000",
        "--learning-rate", "1e-4",
        "--buffer-size", "50000",
        "--batch-size", "256",
        "--gamma", "0.99",
        "--target-network-frequency", "500",
        "--start-e", "1.0",
        "--end-e", "0.05",
        "--exploration-fraction", "0.2",
        "--train-frequency", "4",
        "--save-model",
        "--capture-video",
    ]

    cmd.extend(sys.argv[1:])

    print("=" * 60)
    print("Training DQN on LunarLander-v3")
    print("=" * 60)
    print(f"Script: {DQN_SCRIPT}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

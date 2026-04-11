"""Train PPO on CartPole-v1 using CleanRL's ppo.py.

Week 3 of the RL 101 study group — actor-critic methods.
Calls CleanRL's implementation with tuned defaults for CartPole.
Run: python scripts/train_cartpole_ppo.py

Note: unlike dqn.py, ppo.py does not have --save-model, so there
is no post-training checkpoint to evaluate. Watch training progress
via TensorBoard (`make tensorboard`) or the captured eval videos
in `videos/`.
"""

import subprocess
import sys
from pathlib import Path

PPO_SCRIPT = Path(__file__).resolve().parent.parent / "cleanrl" / "cleanrl" / "ppo.py"


def main():
    if not PPO_SCRIPT.exists():
        print(f"Error: CleanRL not found at {PPO_SCRIPT}")
        print("Run 'make setup' or 'bash setup.sh' first.")
        sys.exit(1)

    cmd = [
        sys.executable, str(PPO_SCRIPT),
        "--env-id", "CartPole-v1",
        "--total-timesteps", "500000",
        "--learning-rate", "2.5e-4",
        "--num-envs", "4",
        "--num-steps", "128",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--num-minibatches", "4",
        "--update-epochs", "4",
        "--clip-coef", "0.2",
        "--ent-coef", "0.01",
        "--vf-coef", "0.5",
        "--capture-video",
    ]

    # Pass through any extra args
    cmd.extend(sys.argv[1:])

    print("=" * 60)
    print("Training PPO on CartPole-v1")
    print("=" * 60)
    print(f"Script: {PPO_SCRIPT}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

# cleanrl-cartpole-dqn

Minimal DQN demo for CartPole-v1 using [CleanRL](https://github.com/vwxyzjn/cleanrl) and [Gymnasium](https://gymnasium.farama.org/). Week 2 homework for the RL 101 study group.

## Quick Start

```bash
make setup          # Create conda env, clone CleanRL, install deps
conda activate rl101
make train          # Train DQN on CartPole-v1 (~5 min, 500K steps)
make eval           # Watch the trained agent balance the pole
```

## What This Demonstrates

**Week 2: Value-Based Methods (Q-learning / DQN)**

| Concept | What it does | Where to see it |
|---------|-------------|-----------------|
| Q-function Q(s,a) | Estimates expected future reward | `dqn.py` — `QNetwork` class |
| Bellman equation | Q(s,a) = r + γ max Q(s',a') | `dqn.py` — TD loss computation |
| Replay buffer | Stores transitions, samples minibatches | `dqn.py` — `ReplayBuffer` |
| Epsilon-greedy | Balance exploration vs exploitation | `dqn.py` — action selection |
| Target network | Stabilizes training targets | `dqn.py` — `target_network` |

## Demo Flow

1. `make random` — Random agent fails in ~10-20 steps
2. `make train` — Train DQN, watch metrics in TensorBoard
3. `make eval` — Trained agent balances for 500 steps (max score)
4. Code walkthrough — Read `cleanrl/cleanrl/dqn.py` together
5. `make tensorboard` — Review training curves

<!-- TODO: Add CartPole before/after GIFs -->

## Commands

```
make help           # Show all available commands
make setup          # Install everything
make random         # Random agent baseline
make train          # Train CartPole DQN
make train-lunar    # Train LunarLander DQN (bonus)
make eval           # Evaluate trained CartPole model
make eval-lunar     # Evaluate trained LunarLander model
make tensorboard    # Launch TensorBoard
make demo           # Full demo: random → train → eval
```

## TensorBoard Metrics

Run `make tensorboard` and watch:
- **charts/episodic_return** — reward per episode (goal: 500)
- **losses/q_values** — Q-value estimates (should increase and stabilize)
- **charts/epsilon** — exploration rate (decays 1.0 → 0.05)
- **losses/td_loss** — TD error (should decrease)

## Requirements

- Python 3.10 (CleanRL requires `<3.11`; setup creates a conda env)
- NVIDIA GPU with CUDA support (tested on RTX 5090)
- PyTorch nightly with CUDA 12.8+
- conda (for environment management)

## Resources

- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file RL implementations
- [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)

---

*RL 101 Study Group — Colby Ziyu Wang @ SparkCraft*

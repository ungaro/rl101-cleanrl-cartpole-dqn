# cleanrl-cartpole-dqn

Week 2 DQN homework demo for the **RL 101** study group (Colby Ziyu Wang @ SparkCraft).

Demonstrates Deep Q-Networks on CartPole-v1 using CleanRL — the simplest way to see
value-based RL in action.

## Tech Stack

- Python 3.10 (CleanRL requires `>=3.8,<3.11`; managed via conda env `rl101`)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file RL implementations
- [Gymnasium](https://gymnasium.farama.org/) — RL environments (CartPole-v1, LunarLander-v3)
- PyTorch nightly (cu128) — required for RTX 5090 (Blackwell SM 12.0)
- TensorBoard — training visualization

## Setup

```bash
# Full setup (creates conda env, clones CleanRL, installs PyTorch nightly + deps)
make setup
# Then activate:
conda activate rl101
```

### Manual Setup Steps

```bash
# 1. Create conda env (CleanRL needs Python <3.11)
conda create -y -n rl101 python=3.10
conda activate rl101

# 2. Clone CleanRL
git clone https://github.com/vwxyzjn/cleanrl.git

# 3. Install PyTorch nightly for RTX 5090 (CUDA 12.8+)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Install CleanRL (--no-deps to avoid pinned torch==2.4.1 conflict)
cd cleanrl && pip install --no-deps -e . && cd ..

# 5. Install runtime dependencies
pip install "gymnasium[classic-control,box2d]" tensorboard tyro wandb moviepy pygame rich numpy
```

## Key Commands

```bash
# Random agent baseline (shows CartPole failing in ~20 steps)
make random

# Train DQN on CartPole-v1 (500K timesteps, ~5 min)
make train

# Train DQN on LunarLander-v3 (1M timesteps, bonus demo)
make train-lunar

# Train PPO on CartPole-v1 (Week 3 — actor-critic, no model save / eval)
make train-ppo

# Evaluate a trained model with rendering
make eval

# Watch training metrics
make tensorboard

# Full demo flow: random → train → eval
make demo
```

## TensorBoard Metrics to Watch

- **charts/episodic_return** — should climb from ~10 to 500 (CartPole max)
- **losses/q_values** — predicted Q-values, should increase and stabilize
- **charts/epsilon** — exploration rate, decays from 1.0 to 0.05
- **losses/td_loss** — temporal difference loss, should decrease over time

## Demo Flow

1. **Random agent** (`make random`) — watch CartPole fail in ~10-20 steps
2. **Train DQN** (`make train`) — train live, watch TensorBoard metrics
3. **Trained agent** (`make eval`) — watch CartPole balance for 500 steps
4. **Code walkthrough** — read through CleanRL's `dqn.py` together
5. **TensorBoard** (`make tensorboard`) — review training curves

## Key DQN Concepts (Week 2)

- **Q-function Q(s,a)** — estimates expected cumulative reward for taking action `a` in state `s`
- **Bellman equation** — Q(s,a) = r + γ max_a' Q(s', a'), the recursive definition of value
- **Replay buffer** — stores (s, a, r, s', done) transitions; samples random minibatches to break correlation
- **Epsilon-greedy** — with probability ε take random action (explore), otherwise take argmax Q (exploit)
- **Target network** — separate, slowly-updated copy of Q-network to stabilize training targets

## Project Structure

```
├── CLAUDE.md              # This file (project context for Claude Code)
├── README.md              # GitHub-facing readme
├── Makefile               # Convenience targets
├── setup.sh               # Environment setup script
├── scripts/
│   ├── random_agent.py    # Random agent baseline
│   ├── train_cartpole.py  # CartPole DQN training wrapper
│   ├── train_lunarlander.py # LunarLander DQN training wrapper
│   └── evaluate.py        # Load and run trained model
├── cleanrl/               # Cloned CleanRL repo (gitignored)
├── runs/                  # TensorBoard logs (gitignored)
└── videos/                # Captured training videos (gitignored)
```

## Important Notes

- **Do NOT modify CleanRL source code.** Treat it as a cloned dependency.
- Training scripts are thin wrappers that call `cleanrl/cleanrl/dqn.py` with good defaults.
- Hardware: RTX 5090 requires PyTorch nightly with CUDA 12.8+ support.

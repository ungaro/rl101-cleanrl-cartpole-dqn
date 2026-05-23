# rl101-crash-course

Multi-week notes and code for the **RL 101** study group (Colby Ziyu Wang
@ SparkCraft, hosted by AI Scholars).

Started as a Week 2 DQN demo on CartPole-v1 using CleanRL; grew with the
course into a full 8-week study group repository: DQN (Week 2), PPO and
Atari (Week 3), agent RL with MiniMax Forge (Week 4), RLHF and the M2.7
case study (Week 5), Reinforcement Fine-Tuning + RLHF deep dive (Week 6),
World Models for LLM Agents / RWML (Week 7), and Robotics Simulation RL
with Isaac Sim & MuJoCo (Week 8) — the last of which also ships three
runnable visual demos (`make demos-help`).

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

### Week 8 Visual Demos

The Week 8 lecture also includes three live demos on the RTX 5090:

```bash
# Demo 1 — Unitree G1 spin kick (mjlab, pretrained ONNX, instant replay)
make setup-mjlab           # install uv + run mjlab sanity check
make setup-spinkick        # clone g1_spinkick_example into external/
make demo-spinkick         # play the pretrained policy in MuJoCo viewer

# Demo 2 — Isaac Lab ANYmal-D rough terrain (live PPO, ~10–15 min)
make setup-isaaclab        # guided install (needs Python 3.11 env)
make demo-anymal           # train
make demo-anymal-play      # visualize

# Demo 3 — Holosoma "humanoid in 15 minutes" (live SAC, ~15 min)
make setup-holosoma        # clone + setup_mujoco_via_uv.sh
make demo-humanoid-15min   # train Unitree G1 from scratch
make demo-humanoid-play    # visualize
```

Full run guide: `docs/week8-demos.md`. Windows helper scripts:
`scripts/holosoma_train_win.py` (Holosoma + Triton/bfloat16 fixes) and
`scripts/train_anymal_win.bat` (Isaac Lab env var setup). External
repos are cloned into `external/` (gitignored). Demos 2 and 3 need
separate conda envs:

- **`rl101-isaac`** — Python 3.11, Isaac Sim 5.1.0, PyTorch 2.7.0+cu128.
- Holosoma's `setup_mujoco_via_uv.sh` creates its own uv-managed env
  inside `external/holosoma/`.

The original **`rl101`** env (Python 3.10, PyTorch nightly cu128) is
still used for Weeks 2–3 CartPole / LunarLander / PPO commands.

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
├── Makefile               # Convenience targets (incl. Week 8 demo wrappers)
├── setup.sh               # Environment setup script (rl101 env)
├── scripts/
│   ├── random_agent.py    # Random agent baseline
│   ├── train_cartpole.py  # CartPole DQN training wrapper
│   ├── train_lunarlander.py # LunarLander DQN training wrapper
│   └── evaluate.py        # Load and run trained model
├── docs/
│   ├── week2-…            # Deep-dive companions (markdown + Marp)
│   ├── week3-…
│   ├── week4-…
│   ├── week5-…
│   ├── week6-…
│   ├── week7-…
│   ├── week8-sim-environments.md  # Robotics sim deep dive
│   └── week8-demos.md     # Runnable demo guide for mjlab / Isaac Lab / Holosoma
├── cleanrl/               # Cloned CleanRL repo (gitignored)
├── external/              # Week 8 demo external repos (gitignored)
│   ├── g1_spinkick_example/
│   ├── IsaacLab/
│   └── holosoma/
├── runs/                  # TensorBoard logs (gitignored)
└── videos/                # Captured training videos (gitignored)
```

## Important Notes

- **Do NOT modify CleanRL source code.** Treat it as a cloned dependency.
- **Do NOT modify external/ repos.** Same convention — mjlab, IsaacLab,
  Holosoma, and g1_spinkick_example are cloned dependencies, not part of
  the repo. The Makefile wrappers in `make demos-help` invoke their
  upstream entry points; full setup lives in `docs/week8-demos.md`.
- Training scripts are thin wrappers that call `cleanrl/cleanrl/dqn.py` with good defaults.
- Hardware: RTX 5090 requires PyTorch nightly with CUDA 12.8+ support
  for the `rl101` env. Isaac Lab v2.3.2 uses stable PyTorch 2.7.0+cu128
  (no nightly needed) in the separate `rl101-isaac` env.
- WSL2: Demos 1 and 3 work cleanly. Demo 2 (Isaac Lab / ANYmal-D) is
  not officially supported on WSL2 — see `docs/week8-demos.md`
  troubleshooting section.

# RL 101 — Study Group Notes & Code

Hands-on notes and code for the **RL 101** study group (Colby Ziyu Wang @ SparkCraft, hosted by AI Scholars).
Each week pairs runnable demos (built on [CleanRL](https://github.com/vwxyzjn/cleanrl)
and [Gymnasium](https://gymnasium.farama.org/)) with a deep-dive write-up that walks
through the math, the algorithm, and the source code line by line.

The repo started as a Week 2 DQN demo and grew with the course. The runnable
demos cover Weeks 2–3 (classic deep RL); Weeks 4–5 are documentation deep-dives
on agent RL and the post-training stack.

## About the Study Group

A 1 hour/week × 6 week **Reinforcement Learning 101** study group — from
basics to RLHF, World Models, and the **MiniMax / Forge** RL-for-agents
framework, with code-throughs of the core RL algorithms. Taught by community
friend **Colby (Ziyu) Wang** (RL class TA at TMU) with 100+ registered
classmates. Open to developers, ML engineers, and ML researchers.

Reinforcement learning matters because it helps AI agents make decisions in
robotics, games, healthcare, and recommendation systems — and is an important
building block on the path toward AGI.

- 📺 Past sessions: [YouTube playlist](https://www.youtube.com/watch?v=4e0laDA7jlM&list=PLte0_KfXCwoh2EX7KRmooLU-Jyn-y8BQZ)
- 🎓 Hosted by **AI Scholars** — a peer-led learning journey for engineers,
  students, researchers, and builders: [aischolars.info](https://aischolars.info/)

## Table of Contents

| Week | Topic | Slides | Deep Dive | Code |
|------|-------|--------|-----------|------|
| 1 | **Introduction to RL** — MDPs, value, policy | [RL 101](https://docs.google.com/presentation/d/1uo6lhYU6gS-AVOiSJHCFMUf3S8H2bnoYMVH7fUd1WoY/edit?usp=sharing) | — | — |
| 2 | **Value-Based Methods — DQN** on CartPole-v1 | [RL 102](https://docs.google.com/presentation/d/1dh3670QGG5a6imEPG_LfzwTtD2_H6dkwjy81uLdLaHo/edit?usp=sharing) | [`docs/week2-dqn-under-the-hood.md`](docs/week2-dqn-under-the-hood.md) | `make train` / `make eval` |
| 3 | **Actor-Critic — PPO** on CartPole-v1 | [RL 103](https://docs.google.com/presentation/d/1CIc8_FcSSqFgSLbFoP_RbD-_gUrqKUuMkXxtKVeb3mc/edit?usp=sharing) | [`docs/week3-ppo-under-the-hood.md`](docs/week3-ppo-under-the-hood.md) | `make train-ppo` |
| 4 | **From Algorithms to Real Systems — MiniMax Forge** | [RL 104](https://docs.google.com/presentation/d/1rcNfBS_MB04ANML4LDUeL67zJBKltBzvP6fk0y6-aJM/edit?usp=sharing) | [`docs/week4-agent-rl-forge.md`](docs/week4-agent-rl-forge.md) | (docs only) |
| 5 | **RLHF and the Path to Agent RL — MiniMax M2.7** | — | [`docs/week5-minimax-m27-visual-guide.md`](docs/week5-minimax-m27-visual-guide.md) | (docs only) |

Each deep-dive renders both as a normal Markdown document on GitHub and as
slides via [marp-cli](https://github.com/marp-team/marp-cli).

## Quick Start (Weeks 2–3 demos)

```bash
make setup          # Create conda env, clone CleanRL, install deps
conda activate rl101

# Week 2 — DQN
make train          # Train DQN on CartPole-v1 (~5 min, 500K steps)
make eval           # Watch the trained agent balance the pole

# Week 3 — PPO
make train-ppo            # Train PPO on CartPole-v1
make train-breakout       # Train PPO on Atari Breakout (10M steps)
make train-pong           # Train PPO on Atari Pong (5M steps)
make train-spaceinvaders  # Train PPO on Atari Space Invaders (10M steps)
```

## Weeks at a Glance

### Week 2 — DQN (Value-Based)

| Concept | What it does | Where to see it |
|---------|--------------|-----------------|
| Q-function Q(s,a) | Estimates expected future reward | `dqn.py` — `QNetwork` class |
| Bellman equation | Q(s,a) = r + γ max Q(s',a') | `dqn.py` — TD loss computation |
| Replay buffer | Stores transitions, samples minibatches | `dqn.py` — `ReplayBuffer` |
| Epsilon-greedy | Balance exploration vs exploitation | `dqn.py` — action selection |
| Target network | Stabilizes training targets | `dqn.py` — `target_network` |

Demo flow: `make random` (random agent fails in ~20 steps) → `make train` →
`make eval` (trained agent balances for 500 steps) → `make tensorboard`.

### Week 3 — PPO (Actor-Critic)

Builds on Week 2 with a policy network alongside the value network, plus
Generalized Advantage Estimation (GAE) and PPO's clipped surrogate objective.
The deep dive walks through CleanRL's `ppo.py` line by line.

Beyond CartPole, Week 3 also includes PPO training on classic Atari
environments (`make train-breakout`, `make train-pong`,
`make train-spaceinvaders`) so you can watch the same algorithm scale
from a 4-dimensional control task to pixel-input arcade games.

### Week 4 — Agent RL with MiniMax Forge

Why RL on LLM agents is different from RL on games or RLHF: 100K+ environments,
minutes-long episodes, 200K-token contexts, partial rollouts. Covers Forge's
three-layer architecture, windowed FIFO scheduling, prefix-tree training,
and the **PPO → GRPO → DAPO → CISPO** algorithmic lineage.

### Week 5 — RLHF and the Path to Agent RL

The post-training pipeline end to end: SFT → reward modeling (Bradley-Terry,
ORM vs PRM) → PPO/GRPO/DAPO → DPO → RLVR → reasoning RL. Concludes with a
MiniMax M2.7 case study tying the pieces together.

## Watching the Trained Agent (Week 2)

After `make train` finishes there are **three ways** to see the trained agent:

### 1. Live pygame window (`make eval`)

Opens a native window showing the cart and pole moving in real time. Uses
`scripts/evaluate.py` with `render_mode='human'`, which automatically loads
the newest model in `runs/`.

```bash
make eval              # 5 episodes with a live window
```

Requires a display:
- **Linux desktop** — works out of the box
- **WSL2 on Windows 11** — works out of the box (WSLg handles it)
- **WSL2 on Windows 10** — needs an X server (e.g. VcXsrv) with `DISPLAY` set
- **Headless/SSH** — won't work; use option 2 or 3 instead

### 2. Auto-captured eval videos (.mp4)

Training runs with `--capture-video`, so CleanRL automatically records **10 eval
episodes** as `.mp4` files at the end of training. No display needed.

```bash
ls videos/CartPole-v1__dqn__1__*-eval/
# rl-video-episode-0.mp4  ...  rl-video-episode-8.mp4

# Open the videos folder in Windows Explorer (from WSL):
explorer.exe "$(wslpath -w videos/)"

# Or on native Linux:
xdg-open videos/
```

### 3. TensorBoard (metrics + inline videos)

```bash
make tensorboard       # then open http://localhost:6006
```

## Commands

```
make help                # Show all available commands
make setup               # Install everything
make random              # Random agent baseline
make train               # Train CartPole DQN (Week 2)
make train-lunar         # Train LunarLander DQN (bonus)
make train-ppo           # Train CartPole PPO (Week 3)
make train-breakout      # Train PPO on Atari Breakout (Week 3, 10M steps)
make train-pong          # Train PPO on Atari Pong (Week 3, 5M steps)
make train-spaceinvaders # Train PPO on Atari Space Invaders (Week 3, 10M steps)
make eval                # Evaluate trained CartPole DQN
make eval-lunar          # Evaluate trained LunarLander model
make tensorboard         # Launch TensorBoard
make demo                # Full demo: random → train → eval
```

## TensorBoard Metrics

Run `make tensorboard` and watch:
- **charts/episodic_return** — reward per episode (CartPole goal: 500)
- **losses/q_values** — Q-value estimates (DQN; should increase and stabilize)
- **charts/epsilon** — exploration rate (DQN; decays 1.0 → 0.05)
- **losses/td_loss** — TD error (DQN; should decrease)
- **losses/policy_loss**, **losses/value_loss**, **losses/entropy** — PPO

## Requirements

- Python 3.10 (CleanRL requires `<3.11`; setup creates a conda env)
- conda (for environment management)
- PyTorch — setup installs the nightly build with CUDA 12.8 for RTX 5090
  (Blackwell SM 12.0); stable PyTorch works fine on older GPUs
- NVIDIA GPU **optional for CartPole** — the Q-network is tiny (~13K params),
  so CPU is typically faster than GPU due to per-step CPU↔GPU transfer
  overhead. Pass `--no-cuda` to `train_cartpole.py` to force CPU. GPU helps
  for LunarLander and larger environments.

## Resources

- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file RL implementations
- [Gymnasium](https://gymnasium.farama.org/) — RL environments
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [MiniMax Forge](https://arxiv.org/abs/2506.13585) — agent RL at scale (Week 4)

---

*RL 101 Study Group — Colby Ziyu Wang @ SparkCraft / Hosted by AI Scholars*
*Notes: Alp Guneysel*

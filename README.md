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

> **Deep dive (Week 2 — DQN):** for a full slide-by-slide walk-through of
> the math, the algorithm, and CleanRL's `dqn.py` line by line, see
> [`docs/week2-dqn-under-the-hood.md`](docs/week2-dqn-under-the-hood.md).
>
> **Deep dive (Week 3 — PPO):** actor-critic methods, GAE, and CleanRL's
> `ppo.py` line by line, see
> [`docs/week3-ppo-under-the-hood.md`](docs/week3-ppo-under-the-hood.md).

## Demo Flow

1. `make random` — Random agent fails in ~10-20 steps
2. `make train` — Train DQN, watch metrics in TensorBoard
3. `make eval` — Trained agent balances for 500 steps (max score)
4. Code walkthrough — Read `cleanrl/cleanrl/dqn.py` together
5. `make tensorboard` — Review training curves

<!-- TODO: Add CartPole before/after GIFs -->

## Watching the Trained Agent

After `make train` finishes there are **three ways** to see the trained agent in action:

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
# rl-video-episode-0.mp4  rl-video-episode-1.mp4  ...  rl-video-episode-8.mp4

# Open the videos folder in Windows Explorer (from WSL):
explorer.exe "$(wslpath -w videos/)"

# Or on native Linux:
xdg-open videos/
```

The later episodes (episode-8, episode-9) show the agent after the eval warmup
and are usually the most interesting to watch.

### 3. TensorBoard (metrics + inline videos)

TensorBoard shows the training curves **and** embeds the captured videos in the
Images tab, so you can scrub through them in your browser without a display.

```bash
make tensorboard       # then open http://localhost:6006
```

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
- conda (for environment management)
- PyTorch — setup installs the nightly build with CUDA 12.8 for RTX 5090
  (Blackwell SM 12.0); stable PyTorch works fine on older GPUs
- NVIDIA GPU **optional for CartPole** — the Q-network is tiny (~13K params),
  so CPU is typically faster than GPU due to per-step CPU↔GPU transfer
  overhead. Pass `--no-cuda` to `train_cartpole.py` to force CPU. GPU helps
  for LunarLander and larger environments.

## Resources

- [CleanRL](https://github.com/vwxyzjn/cleanrl) — single-file RL implementations
- [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)

---

*RL 101 Study Group — Colby Ziyu Wang @ SparkCraft*

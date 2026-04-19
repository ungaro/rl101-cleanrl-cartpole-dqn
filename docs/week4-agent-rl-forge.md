# Week 4: RL in Agents — How MiniMax Forge Trains Agents for Complex Tasks

*From CartPole to 100,000 real-world environments — scaling RL for tool-using,
code-writing, multi-agent systems.*

**Week 4 — RL 101 Study Group**

---

In Weeks 1-3 we trained RL agents on games: CartPole, Atari, MuJoCo. The
environments were fixed, the rewards were clear, and episodes lasted seconds.

This week we tackle a harder question: **how do you train an LLM agent that
writes code, uses tools, browses the web, and coordinates with other agents —
across 100,000+ different environments, with episodes lasting minutes and
contexts spanning 200K tokens?**

MiniMax's **Forge** framework is one of the most detailed public answers to
this question. We'll use it as a lens to understand the engineering and
algorithmic challenges of agent RL at scale.

---

## Table of Contents

### Part I — Why Agent RL Is Hard

1. [From Games to Agents](#1-from-games-to-agents)
2. [The Impossible Triangle](#2-the-impossible-triangle)
3. [What Makes Agent RL Different from RLHF](#3-what-makes-agent-rl-different-from-rlhf)

### Part II — Forge: System Architecture

4. [The Three-Layer Architecture](#4-the-three-layer-architecture)
5. [White-Box vs Black-Box Agents](#5-white-box-vs-black-box-agents)
6. [Windowed FIFO Scheduling](#6-windowed-fifo-scheduling)
7. [Prefix-Tree Merging: 40x Training Speedup](#7-prefix-tree-merging-40x-training-speedup)
8. [Inference Acceleration](#8-inference-acceleration)

### Part III — Forge: The Algorithm

9. [CISPO: Clipped Importance Sampling Policy Optimization](#9-cispo-clipped-importance-sampling-policy-optimization)
10. [Dense and Efficiency-Aware Rewards](#10-dense-and-efficiency-aware-rewards)
11. [Unified Multi-Domain Training](#11-unified-multi-domain-training)

### Part IV — Results and Context

12. [The M2 Family: Same Architecture, Better RL](#12-the-m2-family-same-architecture-better-rl)
13. [Benchmark Performance](#13-benchmark-performance)
14. [Forge vs Other Frameworks](#14-forge-vs-other-frameworks)
15. [What We Learned](#15-what-we-learned)

[Sources](#sources)

---

# Part I — Why Agent RL Is Hard

---

## 1. From Games to Agents

Over the past three weeks we scaled RL from simple to complex:

| Week | Environment | Obs | Actions | Episode | Reward |
|---|---|---|---|---|---|
| 2 | CartPole | 4 floats | 2 | ~500 steps | +1/step (dense) |
| 3 | Breakout | 4x84x84 frames | 4 | ~10K frames | Score (dense) |
| 3 | MuJoCo | 17-376 floats | 3-17 | ~1000 steps | Velocity (dense) |
| **4** | **Agent tasks** | **200K tokens** | **~100K vocab** | **Minutes** | **Task completion (sparse)** |

The jump from Week 3 to Week 4 is qualitative, not just quantitative:

- **Actions are language.** Instead of "move left," the agent generates
  arbitrary text: function calls, code edits, search queries, shell commands.
- **Environments are the real world.** Not simulated physics — actual
  codebases, APIs, web browsers, databases.
- **Episodes have variable length.** A simple API call takes seconds. Debugging
  a complex codebase takes hours with dozens of tool calls.
- **Rewards are sparse.** You often only know at the very end: did the tests
  pass? Did the task complete?

> **Week 3 connection:** The same PPO concepts apply — policy gradients, clipped
> surrogate, GAE. But every component must be re-engineered for this scale.
> Forge is that re-engineering.

---

## 2. The Impossible Triangle

MiniMax frames the core challenge as a **trilemma** — three properties you
need simultaneously but that fight each other:

```
              System Throughput
                    /\
                   /  \
                  /    \
                 /  !!  \
                /________\
   Training               Agent
   Stability              Flexibility
```

**System throughput:** Process millions of tokens per second. Training is
bottlenecked by rollout generation, gradient computation, data I/O, and
inference speed.

**Training stability:** Keep update variance bounded. Prevent distribution
shift from async scheduling. Guarantee convergence.

**Agent flexibility:** Support arbitrary agent architectures — white-box,
black-box, multi-agent, tool-using, context-compressing. Don't couple the
framework to any specific agent design.

### Why they conflict

- **Throughput vs stability:** Async scheduling maximizes GPU utilization but
  causes distribution shift (fast/easy tasks dominate the training batch).
  Sync scheduling prevents shift but wastes compute waiting for stragglers.

- **Throughput vs flexibility:** Supporting arbitrary agents (with variable-
  length episodes, context management, sub-agents) makes scheduling harder
  and introduces unpredictable latency.

- **Stability vs flexibility:** Black-box agents can manipulate context in
  unpredictable ways (compress memory, rewrite history), making the
  token-level training representation inconsistent with what the agent
  actually sees.

Forge's answer: resolve each conflict with a dedicated mechanism. Windowed
FIFO for throughput-stability. Middleware decoupling for flexibility. CISPO
for algorithmic stability.

---

## 3. What Makes Agent RL Different from RLHF

Standard RLHF (the kind that trained ChatGPT) operates in a simple regime:
one prompt, one response, one reward. Agent RL is fundamentally harder:

| | Chat RLHF | Agent RL (Forge) |
|---|---|---|
| Episode | Single turn | Dozens of tool calls |
| Context | 1-2K tokens | Up to 200K tokens |
| Duration | Seconds | Minutes to hours |
| Reward | Preference model (dense) | Task completion (sparse) |
| Actions | Generate text | Text + tool calls + code |
| Environments | One (chat) | 100,000+ scaffolds |
| Latency variance | Low | Extreme (seconds to hours) |

### Three specific challenges Forge addresses

**1. Credit assignment in 200K contexts.**
When an agent writes 50 tool calls across 200K tokens and the task eventually
fails, which action caused the failure? Standard GAE (Week 3) propagates
credit backwards, but across 200K tokens the signal-to-noise ratio becomes
"mathematically precarious." Forge uses **process rewards** (dense
intermediate feedback) and **reward-to-go normalization** to make credit
assignment tractable.

**2. Latency-agnostic optimization.**
Traditional RL optimizes for correctness only. But in agent tasks, a correct
solution that takes 10 minutes is far worse than a correct solution that
takes 30 seconds. Forge adds a **task completion time reward** that
incentivizes parallelism and efficiency.

**3. Context rot.**
As multi-turn conversations grow, accumulated intermediate reasoning and
observations create "attention dilution" — the model loses focus on critical
information even within its context window. Forge treats **context management
as a trainable action**, not just an inference-time hack.

---

# Part II — Forge: System Architecture

---

## 4. The Three-Layer Architecture

Forge's key design principle: **complete decoupling** between the agent's
reasoning logic and the training infrastructure. Three physically separated
layers:

```
┌──────────────────────────────────────────────────────────┐
│  AGENT LAYER                                             │
│                                                          │
│  ┌────────────────┐      ┌─────────────────────────┐    │
│  │   Black-Box    │      │      White-Box           │    │
│  │   Agent        │      │  ┌─────┐  ┌──────────┐  │    │
│  │   (opaque)     │      │  │ LLM │  │ Env      │  │    │
│  └───────┬────────┘      │  └─────┘  └──────────┘  │    │
│          │               │     Agent Loop           │    │
│          │               └────────────┬─────────────┘    │
└──────────┼────────────────────────────┼──────────────────┘
           │                            │
- - - - - -│- - - - - - - - - - - - - - │- - - - - - - - - -
           │                            │
┌──────────v────────────────────────────v──────────────────┐
│  MIDDLEWARE LAYER                                         │
│                                                          │
│  ┌──────────────────┐    ┌───────────────────────────┐  │
│  │  Gateway Server   │    │  Data Pool                │  │
│  │  (completion      │    │  completions: prompt_ids,  │  │
│  │   protocol)       │    │    response_ids            │  │
│  └────────┬──────────┘    │  rewards: outcome, process │  │
│           │               └──────────┬────────────────┘  │
└───────────┼──────────────────────────┼───────────────────┘
            │                          │
- - - - - - │- - - - - - - - - - - - - │- - - - - - - - - -
            │                          │
┌───────────v──────────────────────────v───────────────────┐
│  ENGINES LAYER                                           │
│                                                          │
│  ┌──────────────────┐  sync weights  ┌────────────────┐ │
│  │  Rollout Engine   │ <-----------> │  Train Engine   │ │
│  │  (token gen)      │               │  (CISPO update) │ │
│  └──────────────────┘               └────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Agent Layer

The agent is a **pure trajectory producer**. It focuses on its core logic —
reasoning, tool use, context management — and remains completely agnostic to
the training infrastructure. Both white-box (fully observable) and black-box
(opaque) agents are supported.

### Middleware Layer

The bridge between agents and engines:

- **Gateway Server:** Standardized completion API. Agents send requests here;
  the Gateway routes them to the Rollout Engine. This isolates agents from
  model internals.
- **Data Pool:** Async distributed storage that collects rollout trajectories
  and reward signals. Decouples generation (which is variable-speed) from
  training (which needs consistent batches).

### Engines Layer

- **Rollout Engine:** High-throughput token generation. Responds to Gateway
  requests with model completions.
- **Train Engine:** Consumes batched data from the Data Pool and updates the
  policy using CISPO. Synchronizes weights back to the Rollout Engine.

### Why this matters

The three-layer design means:
- Adding a new agent scaffold requires zero changes to the training pipeline
- The same framework trains across 100,000+ different environments
- Hundreds of scaffold types and thousands of tool invocation formats are
  supported without modification

---

## 5. White-Box vs Black-Box Agents

Forge supports two fundamentally different agent paradigms:

### White-box agents

The framework has full visibility into the agent's state. This enables:

- **Context Management (CM) as a trainable action.** Instead of applying
  context compression only at inference time (which creates a distribution
  shift), Forge models CM as an explicit state transition during training.
  The model learns *when and how* to compress context.
- **CM-driven state transitions.** The state change from $s_t$ to $s_{t+1}$
  encapsulates the context-switching logic, folding context adaptation
  directly into the training objective.
- **Preventing context rot.** The model learns to anticipate context
  management operations, retaining task-critical information while pruning
  noise.

### Black-box agents

The framework knows nothing about the agent's internals. The agent is a
complete black box that sends completion requests through the Gateway.

This supports:
- Proprietary agent architectures
- Complex internal loops (Deep Think, multi-agent)
- Arbitrary context manipulations (memory compression, history rewriting)
- Code-centric agents using Sandbox and MCP environments

**Empirical result:** Even fully opaque black-box agents show consistent
improvement under Forge's RL training — reward increased from ~0.51 to ~0.64
over 55 hours of training in one documented run.

> **The key insight:** During offline evaluation, significant performance
> differences were observed across different scaffolds. The decoupled design
> lets the model train across all scaffolds simultaneously, learning to
> generalize rather than memorize scaffold-specific patterns.

---

## 6. Windowed FIFO Scheduling

The scheduling problem: agent rollouts take wildly different amounts of time.
A simple API call finishes in seconds. A complex debugging session takes hours.
How do you keep GPUs busy without letting the training distribution drift?

### The two bad options

**Strict FIFO (synchronous):** Wait for every task in order. A single slow
task blocks the entire cluster — the "straggler effect." GPU utilization
tanks.

**Greedy async (first-finished-first-out):** Process whatever finishes first.
GPUs stay busy, but the training distribution shifts — early batches are
dominated by fast/easy tasks, late batches by slow/hard tasks. Training
becomes unstable.

### Forge's solution: Windowed FIFO

A sliding window of size $W$ over the generation queue creates a middle
ground:

```
Generation queue:  [T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7]

Window (W=4):      [T_0, T_1, T_2, T_3]  ← scheduler can see these
                                  |
                    T_4, T_5, T_6, T_7   ← invisible, even if done
```

**Three rules:**

1. **Restricted visibility:** The scheduler can only fetch from the window
   $[T_i, T_{i+W-1}]$. Tasks outside are invisible, even if complete.

2. **Local greedy within window:** Inside the window, any completed
   trajectory can be fetched immediately. Fast tasks don't wait for slow
   ones *within* the window. This prevents head-of-line blocking.

3. **Strict blocking at boundary:** Tasks beyond the window are forbidden,
   even if they finished first. This prevents distribution drift toward
   easy tasks.

The window slides forward as head tasks are consumed.

**Concrete parameters** (from the blog):
- Window size: $W = 0.3N$ (30% of generation batch size)
- With batch size $N = 8$: window = 4, max out-of-order tolerance = 3,
  max off-policy lag = 10

> **Week 3 connection:** This is the same off-policy concern we discussed
> with PPO's replay-free design. PPO is on-policy — it discards data after
> each update. Forge can't afford that luxury with 200K-token episodes, so
> Windowed FIFO keeps the off-policy gap bounded.

---

## 7. Prefix-Tree Merging: 40x Training Speedup

Agent training generates many rollout samples that share enormous common
prefixes — the same system prompt, conversation history, retrieved context,
and tool outputs.

### The problem

Naive training treats each sample independently, recomputing the shared
prefix for every sample:

```
Sample A:  [long shared prefix ~~~~~~~~~~~~] [response A]
Sample B:  [long shared prefix ~~~~~~~~~~~~] [response B]
Sample C:  [long shared prefix ~~~~~~~~~~~~] [response C]

→ Shared prefix computed 3x (wasteful)
```

With 200K-token contexts where 90%+ is shared prefix, this is a massive
waste of compute.

### The solution

Prefix-tree merging transforms linear processing into tree-structured
processing:

```
Before:                          After:

A [prefix][resp A]               [prefix] ──┬── [resp A]
B [prefix][resp B]      →                   ├── [resp B]
C [prefix][resp C]                          └── [resp C]

prefix computed 3x               prefix computed 1x
```

**Implementation:** Uses Magi Attention primitives to ensure the tree-
structured forward pass is mathematically identical to independent passes.
After the forward pass, the tree is deconstructed to compute loss normally.
Zero impact on training quality.

**Result:** **40x training speedup** — making 100K+ environment training
economically viable. Also reduces memory overhead, enabling longer sequences
or larger batch sizes.

---

## 8. Inference Acceleration

Three optimizations make Forge's rollout generation fast enough for
production agent RL:

### MTP-based speculative decoding

Uses Multi-Token Prediction heads (the same MTP from M2.7's architecture) as
draft models for speculative decoding. Unlike static draft models, MTP heads
are continuously fine-tuned via Top-K KL loss, ensuring alignment with the
evolving RL policy. This maintains high acceptance rates even as the policy
changes during training.

### Prefill-Decode disaggregation

Separates the prefill phase (processing the prompt) from the decode phase
(generating tokens) onto different hardware. This eliminates interference in
MoE scheduling and allows independent parallelism strategies — maximizing
throughput while optimizing tail latency for long-horizon tasks.

### Global L3 KV Cache Pool

A distributed file system-backed KV cache pool that prevents redundant
prefilling in multi-turn agent RL. A cost-aware scheduler dynamically routes
requests by weighing queuing delay against cache migration costs, maximizing
cache locality without overloading individual instances.

---

# Part III — Forge: The Algorithm

---

## 9. CISPO: Clipped Importance Sampling Policy Optimization

CISPO is Forge's core RL algorithm, designed for the specific challenges of
long-horizon agent training. It evolves PPO (Week 3) in several key ways.

### The problem with PPO for agents

Recall PPO's clipped objective (Week 3):

$$L^{\text{CLIP}} = \min\left(r \cdot A,\  \text{clip}(r, 1-\varepsilon, 1+\varepsilon) \cdot A\right)$$

PPO clips the probability ratio $r$ **symmetrically** around 1. When $r$
leaves $[1-\varepsilon, 1+\varepsilon]$, the gradient is **zero**. For agent
tasks, this is destructive:

- **Discourse tokens** like "wait," "let me reconsider," "actually" often
  change probability rapidly (they're rare in pre-training but critical for
  agent reasoning). PPO permanently zeroes their gradients.
- **Long trajectories** mean more tokens hit the clip boundary, wasting
  gradient signal.

### CISPO's solution: clip weights, not ratios

The CISPO objective:

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i} |o_i|} \sum_{i} \sum_{t} \mathbf{sg}\!\left(\hat{r}_{i,t}\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]$$

where the clipped importance sampling ratio is:

$$\hat{r}_{i,t}(\theta) = \text{clip}\left(r_{i,t}(\theta),\  0,\  1 + \varepsilon^{\text{IS}}_{\text{high}}\right)$$

### Key differences from PPO

| | PPO | CISPO |
|---|---|---|
| Clipping | Symmetric: $[1-\varepsilon, 1+\varepsilon]$ | **One-sided:** $[0, 1+\varepsilon]$ |
| Zero gradients | Yes — tokens outside clip get zero | **No** — all tokens get gradient |
| Gradient through ratio | Yes — ratio is differentiable | **No** — stop-gradient $\mathbf{sg}(\hat{r})$ |
| Critic/value network | Yes | **No** — reward-to-go + baseline |
| Speed reward | No | **Yes** — task completion time |
| Normalization | Per-minibatch advantages | **Token-level** across batch |

**Why one-sided clipping matters:** PPO's lower clip ($1-\varepsilon$) is
what zeros out gradients for tokens that become less likely. CISPO clips at
0 instead — a much softer lower bound. Every token always receives *some*
gradient, just downweighted.

**Why stop-gradient matters:** The ratio $\hat{r}$ acts purely as a weight
on the loss, not as a differentiable quantity. Gradients flow only through
$\log \pi_\theta$ — the policy's log-probability. This is more stable for
MoE models where the ratio can be noisy.

**Empirical result:** 2x faster convergence compared to DAPO on Qwen2.5-32B.

---

## 10. Dense and Efficiency-Aware Rewards

In CartPole, reward design is trivial: +1 per step. In agent RL, it's one
of the hardest problems. Forge uses three complementary reward signals:

### 1. Process rewards (dense intermediate feedback)

Rather than waiting until task completion for a single binary reward,
process rewards score **intermediate behaviors**:

- Penalizing language mixing (switching languages mid-response)
- Penalizing tool invocation errors (wrong API format, missing arguments)
- Rewarding clean reasoning chains
- Scoring each tool call's output quality

> **Week 2 connection:** Process rewards are the agent RL equivalent of
> TD learning — getting feedback at every step rather than waiting for the
> Monte Carlo return at the end. Dense signals guide learning faster.

### 2. Task completion time reward (latency-aware)

A novel signal unique to Forge: the agent's reward includes **relative
completion time**. Faster correct solutions score higher than slower ones.

This incentivizes:
- Parallelizing independent tool calls
- Avoiding redundant steps
- Choosing efficient strategies over thorough-but-slow ones

The advantage estimate combines both:

$$\hat{A}_{i,t} = \sum_{p=t}^{T} (r_p^{\text{speed}} + r_p^{\text{perf}}) - B_i$$

where $B_i$ is a per-prompt baseline for variance reduction.

### 3. Reward-to-go normalization

For 200K-token trajectories, raw returns have enormous variance. Forge uses
**reward-to-go** — the return from each timestep forward — rather than the
full-episode return. Combined with baseline subtraction, this dramatically
reduces gradient variance.

> **Week 3 callback:** This is conceptually the same as GAE — propagating
> credit backwards through the trajectory. The difference is scale: instead
> of 128 steps in CartPole, Forge normalizes across thousands of tokens.

---

## 11. Unified Multi-Domain Training

Unlike sequential training (first reasoning, then QA, then agents), Forge
trains across **three domains simultaneously**:

| Domain | Examples | Why included |
|---|---|---|
| **Reasoning** | Math, logic, planning | Builds thinking backbone |
| **General QA** | Knowledge, conversation | Prevents forgetting |
| **Agent tasks** | Code, tools, multi-agent | The target capability |

**Why simultaneous?** Sequential training causes **negative transfer** —
improving agent capabilities degrades reasoning or general knowledge. Joint
training prevents this by maintaining gradients from all domains.

Short probing runs (500-1000 steps) are used to test domain mixtures before
committing to full training runs.

---

# Part IV — Results and Context

---

## 12. The M2 Family: Same Architecture, Better RL

All M2 models share the **exact same base architecture** — 230B total
parameters, 10B active, 62 layers, 256 experts per layer. The only
difference between versions is post-training:

| Model | Release | Key improvement | SWE-Bench Verified |
|---|---|---|---|
| M2 | 2025 | Initial release | 69.4% |
| M2.1 | 2025 | Improved RL | 74.0% |
| M2.5 | Feb 2026 | Forge + CISPO at scale | **80.2%** |
| M2.7 | Apr 2026 | Self-evolution loop | 56.2% SWE-Pro |

The progression from 69.4% to 80.2% came entirely from **better RL
training** — no architectural changes. This is a powerful demonstration
that post-training (specifically agent RL) can deliver frontier performance
on a fixed architecture.

**Cost efficiency:** M2.5 matches Claude Opus 4.6 speed at roughly 1/10th
the cost ($0.30/M input tokens, $2.40/M output tokens). Running
continuously for one hour costs approximately $1.

---

## 13. Benchmark Performance

### M2.5 results

| Benchmark | M2.5 Score | Context |
|---|---|---|
| SWE-Bench Verified | **80.2%** | Software engineering |
| Multi-SWE-Bench | 51.3% | Multi-domain programming |
| BrowseComp | 76.3% | Web browsing tasks |

### M2.7 results (full table)

| Benchmark | M2.7 Score | Context |
|---|---|---|
| SWE-Pro | 56.22% | Matches GPT-5.3 Codex |
| VIBE-Pro | 55.6% | End-to-end project delivery |
| Terminal Bench 2 | 57.0% | Complex engineering |
| SWE Multilingual | 76.5 | Real-world multilingual |
| GDPval-AA ELO | 1495 | Professional domain expertise |
| Toolathon | 46.3% | Tool interaction |
| MM Claw | 62.7% | Professional work tasks |
| MLE Bench Lite | 66.6% avg | ML competitions |

---

## 14. Forge vs Other Frameworks

Forge is not the only agent RL framework. Here's how it compares:

| | Forge (MiniMax) | OpenRLHF | veRL (ByteDance) | TRL (HF) |
|---|---|---|---|---|
| Focus | Agent RL at scale | General RLHF | Flexible RL training | Easy-to-use RLHF |
| Agent support | White-box + black-box | Limited | Limited | Minimal |
| Scheduling | Windowed FIFO | Standard | Standard | Standard |
| Prefix optimization | Tree merging (40x) | No | No | No |
| Algorithm | CISPO | PPO, REINFORCE++, GRPO | PPO, GRPO, DAPO | PPO, GRPO, DPO |
| Scale tested | 200K+ envs, 230B model | 70B+ | 671B | Small-medium |
| Open-source | No (proprietary) | Yes | Yes | Yes |

Forge's main advantages are agent-specific: the middleware decoupling, prefix
tree merging, and CISPO's speed-aware rewards. Its main disadvantage is that
it's proprietary — you can't use it yourself. For open-source agent RL, the
closest options are OpenRLHF and veRL.

---

## 15. What We Learned

### The evolution of RL in this course

| Week | What we optimized | Reward source | Scale |
|---|---|---|---|
| 2 | CartPole balance | Environment (+1/step) | 4 obs, 2 actions |
| 3 | Atari games, robot control | Game score, velocity | 84x84 frames, 6 actions |
| 4 | Code writing, tool use, agents | Task completion + speed | 200K tokens, 100K vocab |

The algorithms evolved — DQN → PPO → CISPO — but the core loop stayed the
same: **act, observe reward, compute advantage, update policy**.

### Key takeaways from Forge

1. **Architecture matters as much as algorithms.** Forge's 40x speedup from
   prefix-tree merging is purely an engineering optimization — it doesn't
   change the math, but it makes agent RL economically viable.

2. **Speed is a reward signal.** Traditional RL only optimizes for
   correctness. Adding completion time as a reward produces agents that are
   both correct and efficient.

3. **Decoupling is the key to flexibility.** The three-layer middleware
   design lets Forge train across 100,000+ environments without any
   framework changes. The agent is a black box; the framework doesn't care
   how it works.

4. **Post-training > architecture.** The M2 family proves that the same
   230B-parameter model goes from 69% to 80% on SWE-Bench purely through
   better RL training. No architecture changes needed.

5. **Context management must be trained, not hacked.** Applying context
   compression only at inference time creates distribution shift. Forge
   treats it as a trainable action.

---

## Sources

**MiniMax Forge:**
- MiniMax, [*Forge: Scalable Agent RL Framework and Algorithm*](https://huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm) (HuggingFace blog, February 2026) — primary technical reference
- MiniMax, [*Forge: Scalable Agent RL Framework*](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) (official blog, February 2026)

**MiniMax models:**
- [MiniMax-M2.5](https://github.com/MiniMax-AI/MiniMax-M2.5) (GitHub) — model card, architecture details
- MiniMax-M1 CISPO paper: [arXiv:2506.13585](https://arxiv.org/abs/2506.13585)
- MiniMax-01 architecture: [arXiv:2501.08313](https://arxiv.org/abs/2501.08313)

**Open-source agent RL frameworks:**
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — PPO, REINFORCE++, GRPO for LLMs
- [veRL](https://github.com/volcengine/verl) — ByteDance's flexible RL training framework

---

*RL 101 Study Group — Week 4*
*Colby Ziyu Wang @ SparkCraft*

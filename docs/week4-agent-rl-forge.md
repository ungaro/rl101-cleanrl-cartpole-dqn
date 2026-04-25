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

9. [The PPO → GRPO → DAPO → CISPO Evolution](#9-the-ppo--grpo--dapo--cispo-evolution)
10. [CISPO: Clipped Importance Sampling Policy Optimization](#10-cispo-clipped-importance-sampling-policy-optimization)
11. [Dense and Efficiency-Aware Rewards](#11-dense-and-efficiency-aware-rewards)
12. [Unified Multi-Domain Training](#12-unified-multi-domain-training)

### Part IV — Results and Context

13. [Architectural Context: Lightning Attention and MoE](#13-architectural-context-lightning-attention-and-moe)
14. [The M2 Family: Same Architecture, Better RL](#14-the-m2-family-same-architecture-better-rl)
15. [Benchmark Performance](#15-benchmark-performance)
16. [Forge vs Other Frameworks](#16-forge-vs-other-frameworks)
17. [What We Learned](#17-what-we-learned)

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

- **Gateway Server:** Standardized completion API modeled after OpenAI's
  `chat/completions`. Agents send `{prompt_ids, sampling_params}` here; the
  Gateway routes to a rollout replica and returns `{response_ids, logprobs,
  tool_call_metadata}`. The protocol is stateless — agents carry their own
  conversation state.
- **Data Pool:** Distributed async buffer with two sub-stores:
  - `completions[]`: `{prompt_ids, response_ids, sampling_logprobs, episode_id, turn_idx}`
  - `rewards[]`: `{outcome_reward, process_reward[], completion_time}`

  This decouples generation (variable-speed, bursty) from training (needs
  consistent batches). It also enables experience reuse — the same trajectory
  can feed multiple gradient updates via importance sampling.

### Engines Layer

- **Rollout Engine:** High-throughput token generation. Serves Gateway
  requests via a fleet of inference replicas (vLLM/SGLang-class).
- **Train Engine:** Pulls batches from Data Pool, computes CISPO gradients,
  updates the policy. Broadcasts new weights to all rollout replicas.

### The 4-interface agent API

From MiniMax's post-training report, every agent integrates via exactly four
functions:

```python
agent_reprocess(raw_trajectory) -> tokens    # format-specific → canonical tokens
agent_run(state, gateway) -> trajectory      # the agent's core loop
agent_postprocess(trajectory) -> samples     # split into training examples
calculate_reward(trajectory, env) -> rewards # dense + outcome signals
```

This is the entire surface area. A new scaffold (React, Tree-of-Thoughts,
multi-agent team, CodeAct) plugs in by implementing these four functions —
the rest of the pipeline is scaffold-agnostic.

### Weight synchronization

The Rollout-Train split creates an unavoidable off-policy gap: rollouts are
generated by policy $\pi_{\theta_{\text{old}}}$ while updates target $\pi_\theta$.
Forge bounds this gap via:

1. **NCCL broadcast** after each update pushes new weights to all rollout
   replicas (on the order of seconds for a 230B MoE model with expert sharding).
2. **Windowed FIFO** (Section 6) caps how stale the oldest in-flight rollout
   can be — in their configuration, max off-policy lag = 10 updates.
3. **CISPO's importance sampling** (Section 10) mathematically corrects the
   residual distribution mismatch at each token.

### Why this matters

The three-layer design means:
- Adding a new agent scaffold requires zero changes to the training pipeline
- The same framework trains across 100,000+ different environments
- Hundreds of scaffold types and thousands of tool invocation formats are
  supported without modification
- **Throughput breakdown:** MiniMax reports ~60% of cluster compute goes to
  rollout generation, ~40% to training — a ratio that's only possible because
  generation and training can progress asynchronously via the Data Pool.

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

### The off-policy lag math

With generation batch $N$ and window $W$, Forge can derive two hard bounds:

$$\text{max out-of-order tolerance} = W - 1$$

$$\text{max off-policy lag} = N + (W - 1) - 1$$

**Worked example** (from the Forge blog, $N=8$, $W=4$):
- Window contains 4 tasks at a time.
- Within the window, up to $W-1 = 3$ tasks can be fetched out of order.
- A trajectory can lag up to $8 + 3 - 1 = 10$ policy updates behind the
  current training step — but never more.

This bound is what makes CISPO's importance sampling correction tractable:
the ratio $r = \pi_\theta / \pi_{\theta_{\text{old}}}$ never explodes because
$\theta_{\text{old}}$ is never more than 10 updates stale.

### Tuning W

- $W \to 1$: Strict synchronous FIFO. Zero distribution drift, maximum
  straggler waste. GPU utilization plummets on long-tail episodes.
- $W \to N$: Fully greedy async. Max throughput, maximum drift — training
  batches become biased toward whichever tasks happen to finish fast.
- $W = 0.3N$ (Forge's choice): Empirically balances the two. The 30%
  slack absorbs stragglers without letting the policy gap grow unbounded.

> **Week 3 connection:** This is the same off-policy concern we discussed
> with PPO's replay-free design. PPO is on-policy — it discards data after
> each update. Forge can't afford that luxury with 200K-token episodes, so
> Windowed FIFO keeps the off-policy gap bounded, and CISPO's importance
> sampling absorbs whatever gap remains.

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
waste of compute. In realistic agent datasets, the *Potential Overlap
Ratio* (POR) — the fraction of tokens that could be deduplicated — runs
from **28% to 89%** depending on task type.

### The solution: tree-structured forward pass

Prefix-tree merging transforms linear processing into tree-structured
processing:

```
Before:                          After:

A [prefix][resp A]               [prefix] ──┬── [resp A]
B [prefix][resp B]      →                   ├── [resp B]
C [prefix][resp C]                          └── [resp C]

prefix computed 3x               prefix computed 1x
```

### How Magi Attention makes this exact

The tricky part isn't packing the tree — it's proving the packed forward
and backward passes produce *identical* gradients to the naive version.
Three mechanisms make it work:

**1. Shared-prefix attention mask.** A block-sparse mask where branch $i$
attends to:
- the shared prefix tokens (all of them), and
- only its own branch tokens.

Branches cannot attend to each other, so branch B can't leak into
branch A's computation.

**2. Position-ID restoration.** Positional encodings (RoPE) are computed
using each token's position *in its original sequence*, not its position
in the packed buffer. Without this, the prefix token at absolute position
50 would get RoPE embedding 50 in branch A but 200 in branch B — breaking
equivalence.

**3. Gradient restoration in the backward pass.** The shared prefix's
gradient must equal the sum of gradients from all child branches:

$$\frac{\partial \mathcal{L}}{\partial x_{\text{prefix}}} = \sum_{b \in \text{branches}} \frac{\partial \mathcal{L}_b}{\partial x_{\text{prefix}}}$$

The naive tree-packed backward pass computes only one branch's
contribution. Magi Attention's custom FlashAttention-V3 kernel scales the
prefix gradient by the branching factor — a constant-time fix that restores
exact equivalence.

### Result

- **~40x training speedup** in multi-turn agent RL (per MiniMax).
- Independent academic work ([Tree Training, 2511.00413](https://arxiv.org/abs/2511.00413))
  reports 5.7x ideal / 3.9x end-to-end on similar workloads — suggesting
  MiniMax's 40x number is against a much weaker baseline (likely naive
  per-sample training with no cache reuse).
- Memory savings scale linearly with POR, enabling either larger batches
  or longer contexts at the same GPU budget.

> **Why exact equivalence matters:** A 40x speedup that changes the
> gradient is just a different algorithm with different convergence
> properties. The exact-equivalence guarantee means you can train with
> Magi Attention and still reason about convergence using standard PG
> theory.

---

## 8. Inference Acceleration

Three optimizations make Forge's rollout generation fast enough for
production agent RL:

### MTP-based speculative decoding

**Background: speculative decoding.** A small "draft" model proposes $k$
tokens at once; the main model verifies them in a single forward pass. If
accepted, you got $k$ tokens for the price of one. Acceptance rate
determines the speedup.

**The draft-model problem for RL.** Traditional speculative decoding uses
a *static* draft (e.g., a smaller distilled model). During RL training,
the main policy changes every few minutes — the static draft's
distribution drifts away from the policy, and acceptance rate collapses.

**MiniMax's fix — MTP heads trained online.** Multi-Token Prediction
(originally from DeepSeek-V3) adds $D$ lightweight transformer blocks to
the main model, each predicting $d$ tokens ahead. These MTP heads serve
as the draft.

The standard MTP training loss is cross-entropy weighted with $\lambda$
decaying from 0.3 to 0.1:

$$\mathcal{L}_{\text{MTP}} = \sum_{d=1}^{D} \lambda_d \cdot \text{CE}(p_d^{\text{MTP}}, y_{t+d})$$

**Forge's innovation: Top-K KL loss.** Cross-entropy only matches the
argmax token — but spec decoding accepts based on distribution agreement
across the top-$k$ candidates. Forge replaces CE with a Top-K KL objective:

$$\mathcal{L}_{\text{Top-K KL}} = \sum_{d=1}^{D} \text{KL}\!\left(\text{Top}_k(p_d^{\text{MTP}}) \,\|\, \text{Top}_k(p_d^{\text{main}})\right)$$

This directly optimizes for the acceptance rate rather than using CE as a
proxy. Acceptance stays high even as the policy evolves, because the MTP
heads track the main model's *distribution*, not just its top-1 choice.

Combined with spec decoding, reported throughput is 1.5–2x over
autoregressive generation during RL rollouts.

### Prefill-Decode (PD) disaggregation

**The problem.** Prefill (processing the prompt) is compute-bound —
thousands of tokens in one massive matrix multiply. Decode (generating one
token at a time) is memory-bound — repeatedly reading the KV cache. On the
same GPU, they interfere: a prefill burst starves decode requests,
spiking latency.

**The fix.** Physically split prefill and decode onto separate GPU pools,
each with its own parallelism strategy:

| Phase | Workload | Typical parallelism |
|---|---|---|
| Prefill | Bulk matmul, high FLOP/memory | Tensor + pipeline parallel |
| Decode | KV-cache scan, low FLOP/memory | Data parallel (many replicas) |

For MoE models this matters more: prefill and decode trigger very different
expert-routing patterns, and colocating them causes expert-hotspot pileups.
Disaggregation lets each phase use the expert-parallelism scheme that fits
its traffic profile.

> **Ecosystem note:** PD disaggregation was popularized by
> [DistServe (OSDI 2024)](https://arxiv.org/abs/2401.09670) and is now
> standard in vLLM, SGLang, NVIDIA Dynamo, and MoonCake. Forge's
> contribution is its MoE-specific routing for 230B-class models.

### Global L3 KV Cache Pool

Multi-turn agent rollouts repeatedly send near-identical prefixes (previous
turns + tool outputs). A single-replica KV cache only helps if the request
lands on the same replica that saw the prefix before.

Forge's L3 pool stores KV caches in a distributed filesystem accessible to
all rollout replicas. A cost-aware scheduler routes each request by
minimizing:

$$\text{cost}(r, \text{replica}_j) = \text{queuing\_delay}_j + \alpha \cdot \text{cache\_migration\_cost}(r, j)$$

Cheap local hits win; expensive migrations happen only when the local
replica is saturated. This keeps the effective cache hit rate high even
under load imbalance.

---

# Part III — Forge: The Algorithm

---

## 9. The PPO → GRPO → DAPO → CISPO Evolution

Before diving into CISPO's full objective, it helps to see how it emerged.
Each step in this lineage fixes a specific failure of the previous one, and
understanding the chain makes CISPO's design choices obvious rather than
arbitrary.

All four algorithms share the same skeleton: a policy-gradient update
weighted by an importance-sampling ratio and an advantage estimate. What
changes is how each handles the *three structural problems* that appear
when you scale to long-horizon LLM training:

1. **Credit assignment** — how advantage $\hat{A}$ is estimated.
2. **Variance** — how the objective prevents high-variance updates from
   blowing up training.
3. **Gradient survivability** — which tokens receive non-zero gradient.

### 9.1 PPO (2017, Schulman et al.)

The starting point. Week 3's algorithm:

$$\mathcal{L}^{\text{PPO}} = \mathbb{E}\left[ \min\!\left(r_{t}\hat{A}_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\,\hat{A}_t\right) \right]$$

- **Advantage:** Learned critic + GAE.
- **Variance control:** Symmetric clipping zeroes gradients outside
  $[1-\varepsilon, 1+\varepsilon]$.
- **Per-token gradient:** Zero if the ratio leaves the band.

Problems at LLM scale:
- Critic is another 100B+ network → 2x memory, 2x training instability.
- The lower clip $(1-\varepsilon)$ permanently kills tokens whose
  probability drops — often these are exactly the exploratory tokens you
  *want* the policy to learn.
- PPO needs on-policy data; throwing away stale trajectories is prohibitive
  when a single episode is 200K tokens.

### 9.2 GRPO (Feb 2024, DeepSeekMath — arXiv:2402.03300)

**Fix #1: Drop the critic.** Instead of learning $V(s)$, sample $G$
completions per prompt and standardize rewards *within the group*:

$$\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G) + \epsilon}$$

Every token in completion $i$ gets the same $\hat{A}_i$. The advantage
signal comes from "is this completion better than its siblings" rather
than "is this state better than what my critic expects."

Full objective (identical clipping to PPO, but with KL as an *added*
penalty rather than baked into the reward):

$$\mathcal{L}^{\text{GRPO}} = -\min\!\left(r_t \hat{A}, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}\right) + \beta_{\text{KL}}\,\text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**DeepSeekMath hyperparameters:** $G=64$, $\varepsilon=0.2$,
$\beta_{\text{KL}}=0.1$, batch = 1024 (16 prompts × 64 completions).

**Wins:** ~50% memory reduction (no critic), cleaner variance via group
normalization, works on any verifiable reward without needing a shaped
value function.

**Still broken:** Symmetric clipping still zeroes exploratory-token
gradients. The KL penalty is often too restrictive for long chain-of-thought.

### 9.3 DAPO (Mar 2025, ByteDance — arXiv:2503.14476)

**"Decoupled Clip and Dynamic sAmpling Policy Optimization"** — four
surgical fixes to GRPO for long-CoT RL:

**Fix #2a: Clip-Higher.** Decouple the upper and lower clip bounds:

$$\text{clip}(r_t, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}), \quad \varepsilon_{\text{low}}=0.2,\ \varepsilon_{\text{high}}=0.28$$

Raising the upper bound lets low-probability exploratory tokens ramp up
faster without hitting the clip ceiling — fixes entropy collapse.

**Fix #2b: Dynamic sampling.** Drop prompts where all $G$ completions are
correct (accuracy = 1) or all wrong (accuracy = 0). These produce
zero-variance groups → zero advantage → zero gradient. Wasted batch.

**Fix #2c: Token-level loss aggregation.** GRPO averages the loss per
sample first, then across samples — so a 10K-token correct trajectory and
a 100-token wrong one weight equally. DAPO aggregates at the token level:

$$\frac{1}{\sum_i |o_i|} \sum_{i,t} \ell_{i,t}$$

Long trajectories get proportionally more gradient mass.

**Fix #2d: Overlong reward shaping + drop the KL penalty.** Soft-penalize
truncated-at-max-length outputs; remove the explicit KL term (it prevents
the model from diverging enough for long-CoT).

**DAPO hyperparameters:** LR=1e-6, rollout batch=512, $G=16$, max gen
length=20,480. **Result:** 50 pts on AIME 2024 (Qwen2.5-32B), vs 47 for
DeepSeek-R1-Zero-Qwen-32B, using 50% of its training steps.

### 9.4 CISPO (Jun 2025, MiniMax — arXiv:2506.13585)

**Fix #3: Stop clipping the tokens — clip the *weights*.**

DAPO still has the fundamental problem that *some* tokens get zero gradient
(when the ratio leaves the asymmetric band). MiniMax observed this
empirically kills discourse tokens — "wait," "however," "let me reconsider"
— which are rare in pre-training but essential for reasoning.

CISPO's insight: the clip isn't there to zero-out the gradient; it's there
to bound the *importance-sampling weight*. Decouple those two roles by:

1. **Clipping the ratio** as a weight coefficient on the loss.
2. **Never zeroing** any token's gradient — every token contributes through
   $\log \pi_\theta$.
3. **Stop-gradient on the ratio** so it truly behaves as a weight, not as
   a differentiable thing that backprops into the policy twice.

The next section spells out the full objective.

### Evolution at a glance

| | Critic | Advantage | Clipping role | Zero gradients? | KL |
|---|---|---|---|---|---|
| PPO | Yes (GAE) | $\hat{A}_t$ per token | Zero tokens outside band | Yes (symmetric) | In reward |
| GRPO | No | Group-standardized | Zero tokens outside band | Yes (symmetric) | Penalty term |
| DAPO | No | Group + token-level loss | Zero tokens outside asymmetric band | Yes (asymmetric) | Dropped |
| CISPO | No | Group + token-level | **Weight on loss only** | **No** | Dropped |

---

## 10. CISPO: Clipped Importance Sampling Policy Optimization

### The full objective

$$\mathcal{J}_{\text{CISPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o_i\}\sim\pi_{\theta_{\text{old}}}}\!\left[\frac{1}{\sum_i |o_i|} \sum_i \sum_t \mathbf{sg}\!\left(\hat{r}_{i,t}\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})\right]$$

where the clipped IS weight is:

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon^{\text{IS}}_{\text{low}},\ 1 + \varepsilon^{\text{IS}}_{\text{high}}\right)$$

and $\mathbf{sg}(\cdot)$ is the stop-gradient operator.

### Dissecting the objective, term by term

**The $\log \pi_\theta(o_{i,t} \mid \cdot)$ factor.** This is the *only*
place where $\theta$ appears differentiably. Every gradient contribution
flows through it. Unlike PPO/GRPO/DAPO, CISPO looks like a weighted
REINFORCE objective: $\sum \text{weight} \cdot \log \pi_\theta$.

**The $\mathbf{sg}(\hat{r}_{i,t})$ factor.** The clipped importance ratio
acts as a *scalar coefficient*. Stop-gradient means backprop treats it as
a constant — gradients don't flow *into* the ratio computation. This
matters for MoE models where the ratio involves expert routing, which can
be extremely noisy; without $\mathbf{sg}$, that noise would inject
second-order derivatives into every parameter update.

**The $\hat{A}_{i,t}$ factor.** For MiniMax-M1, this is the group-relative
advantage from GRPO:

$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$

same $\hat{A}_i$ across all tokens in a response. For Forge (in the M2.5
training run), it's replaced with a reward-to-go formulation to handle
200K-token agent trajectories:

$$\hat{A}_{i,t} = \sum_{p=t}^{T} (r_p^{\text{speed}} + r_p^{\text{perf}}) - B_i$$

where $B_i$ is a per-prompt baseline. This token-level decomposition gives
finer-grained credit assignment for long episodes (see Section 11).

**The $1 / \sum_i |o_i|$ normalization.** Token-level, inherited from DAPO.
Long trajectories get proportional influence; short ones don't dominate.

### Asymmetric clipping — effectively one-sided

From the MiniMax-M1 paper:

> "We did not impose a lower bound on the IS weight by setting
> $\varepsilon^{\text{IS}}_{\text{low}}$ to a large value; instead, we only
> tuned $\varepsilon^{\text{IS}}_{\text{high}}$."

So in practice $1 - \varepsilon^{\text{IS}}_{\text{low}} \approx 0$ — the
lower clip is disabled. The effective range is $[0, 1 + \varepsilon_{\text{high}}]$.

**Why this works:** The lower clip in PPO exists to prevent a single token
with a vanishingly-small new probability from being upweighted too
aggressively in a "pessimistic minimum." But in CISPO, $\hat{r}$ is a
*weight* not a gradient multiplier in the destructive sense — if $\hat{r}
\to 0$ for a token, its contribution just becomes zero *in magnitude*,
not zero *in direction*. No harm done.

### Hyperparameters (from the MiniMax-M1 paper)

| Parameter | Value |
|---|---|
| Optimizer | AdamW, $\beta_1=0.9$, $\beta_2=0.95$, $\text{eps}=10^{-15}$ |
| Off-policy updates per batch | 16 |
| $\varepsilon^{\text{IS}}_{\text{high}}$ | Tuned (exact value not disclosed) |
| $\varepsilon^{\text{IS}}_{\text{low}}$ | Set large — effectively disabled |
| KL penalty | **None** |
| Group size $G$ | Shared with GRPO-family (typically 16–64) |
| Advantage normalization | Group-level standardization |

### Empirical result

- On Qwen2.5-32B with AIME 2024, CISPO **matches DAPO's peak score in
  half the training steps** — effectively a 2x compute speedup.
- On MiniMax-M1 (full training), CISPO + hybrid attention completed RL
  training in 3 weeks at $534,700 total rental cost.

### Known failure modes

The M1 paper documents two CISPO-adjacent pathologies seen during scaling:

**Length bias.** Generative reward models preferred longer outputs
regardless of actual reasoning quality. Mitigation: length-normalized
reward models, penalty on redundant text.

**Pattern collapse.** Once output length crossed a threshold, trailing
tokens degenerated into incoherent repetition. Root cause: disproportionately
large negative gradients from low-probability tokens at the tail.
Mitigations:
- **Early truncation:** halt generation if 3,000 consecutive tokens have
  probability > 0.99 (sign the model has entered a repetition loop).
- **Tighter gradient clipping** on low-probability tokens at the tail.
- **Loss normalization** across trajectory length.

> **Week 3 callback:** PPO's clipping was a variance-reduction hack. GRPO's
> critic removal was a memory hack. DAPO's asymmetric clip was an
> exploration hack. CISPO's stop-gradient-on-weight is the first of these
> that changes the *structure* of the objective rather than tweaking a
> knob — it turns the update from "clipped surrogate minimum" into
> "importance-weighted REINFORCE." That's why it's a new algorithm, not a
> PPO variant.

---

## 11. Dense and Efficiency-Aware Rewards

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

## 12. Unified Multi-Domain Training

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

## 13. Architectural Context: Lightning Attention and MoE

CISPO and Forge don't exist in a vacuum — they're tuned for MiniMax's
specific model architecture. Two design choices shape how the algorithm
behaves at scale.

### Lightning Attention — 7-out-of-8 layers

All M2-family models inherit the hybrid attention scheme from
[MiniMax-01](https://arxiv.org/abs/2501.08313):

```
Layer pattern (repeating 8-layer block):
  L1–L7: Lightning Attention (linear, hardware-tiled)
  L8:    Softmax attention
```

**Lightning Attention** is a hardware-aware implementation of linear
attention: $O(N)$ in sequence length instead of $O(N^2)$, with tiled
computation that stays inside L2/L3 cache. Memory footprint is nearly
flat across context windows — which is what makes 200K-token agent
rollouts affordable.

The 1-in-8 softmax layers exist because pure linear attention degrades on
certain recall patterns. The hybrid recovers full-attention quality while
keeping 7/8 of layers at linear cost.

**Why this matters for RL:** CISPO's 16 off-policy updates per batch would
be infeasible with quadratic attention on 200K contexts. Lightning
attention is what makes long-context RL *economical*, not just possible.

### MoE — 230B total, 10B active

M2.5's forward pass activates only 10B of 230B total parameters (the top-k
experts chosen by the router). This creates specific challenges for RL:

1. **Router noise in importance ratios.** When expert routing differs
   between $\pi_{\theta_{\text{old}}}$ and $\pi_\theta$, the same token
   can have wildly different probabilities. Stop-gradient on the ratio
   (CISPO's $\mathbf{sg}(\hat{r})$) stops this noise from polluting the
   policy gradient.

2. **Expert load imbalance under RL.** Certain reasoning patterns route
   to specific experts; RL that concentrates on those patterns starves
   others. Forge mitigates this via unified multi-domain training
   (Section 12) — the domain diversity keeps all experts active.

3. **Prefill vs decode routing divergence.** Prefill activates experts in
   a sparse, one-shot pattern; decode activates them repeatedly in a
   narrow loop. PD-disaggregation (Section 8) lets each phase use the
   expert-parallelism strategy that matches its traffic.

### Putting it together

The system works *because* the pieces fit:

- Lightning attention → affordable 200K contexts → agent RL possible
- MoE → efficient parameter scaling → 230B model in 10B compute
- CISPO's stop-gradient → MoE ratio noise absorbed
- Windowed FIFO → off-policy lag bounded → CISPO's IS correction tractable
- Magi Attention → shared-prefix compute reused → training is 40x cheaper
- MTP spec decoding + PD disagg → rollout generation keeps up with training

No single piece is revolutionary. The system is revolutionary.

---

## 14. The M2 Family: Same Architecture, Better RL

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

## 15. Benchmark Performance

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

## 16. Forge vs Other Frameworks

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

## 17. What We Learned

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

6. **Clip weights, not gradients.** The PPO → GRPO → DAPO → CISPO lineage
   shows four generations of researchers realizing that the clipping in
   "clipped surrogate objective" was serving two roles — bounding IS
   weights and zeroing outlier gradients — and that these should be
   decoupled. CISPO's stop-gradient-on-ratio is the cleanest split: every
   token contributes gradient, the ratio is just a weight.

7. **No component is novel; the system is.** Lightning attention, MoE,
   MTP spec decoding, PD disaggregation, GRPO-style advantages, KV cache
   pools — every technique in Forge exists independently in published work.
   Forge's contribution is the co-design: each piece is chosen so that
   the weaknesses of one are absorbed by the strengths of another.

---

## Sources

**MiniMax Forge (primary):**
- MiniMax, [*Forge: Scalable Agent RL Framework and Algorithm*](https://huggingface.co/blog/MiniMax-AI/forge-scalable-agent-rl-framework-and-algorithm) (HuggingFace blog, February 2026) — primary technical reference
- MiniMax, [*Forge: Scalable Agent RL Framework*](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) (official blog, February 2026)
- MiniMax, [*Post-Training Experience and Insights for Agent Models*](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models) — source of the 4-interface API (`agent_reprocess`, `agent_run`, `agent_postprocess`, `calculate_reward`) and training-inference consistency findings

**MiniMax models and CISPO:**
- MiniMax-M1 paper (CISPO algorithm): [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) — full CISPO derivation, hyperparameters, 2x-over-DAPO benchmark
- MiniMax-01 paper (Lightning Attention, MoE): [arXiv:2501.08313](https://arxiv.org/abs/2501.08313)
- [MiniMax-M2.5](https://github.com/MiniMax-AI/MiniMax-M2.5) (GitHub) — model card, architecture details

**Algorithmic lineage (PPO → GRPO → DAPO → CISPO):**
- PPO: Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017)
- GRPO: DeepSeek, [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning*](https://arxiv.org/abs/2402.03300) (Feb 2024)
- DAPO: ByteDance + Tsinghua, [*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*](https://arxiv.org/abs/2503.14476) (Mar 2025) — decoupled clip, dynamic sampling, token-level loss
- CISPO: see MiniMax-M1 paper above

**Infrastructure references:**
- MTP (Multi-Token Prediction): [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (Dec 2024) — MTP heads and spec decoding
- Prefix-tree for RL (independent academic work): [*Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse*](https://arxiv.org/abs/2511.00413) (2025)
- PD disaggregation: [DistServe (OSDI 2024)](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677), [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

**Open-source agent RL frameworks:**
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — PPO, REINFORCE++, GRPO for LLMs
- [veRL](https://github.com/volcengine/verl) — ByteDance's flexible RL training framework (implements DAPO, GRPO)
- [DAPO reference implementation](https://github.com/BytedTsinghua-SIA/DAPO)

---

*RL 101 Study Group — Week 4*
*Colby Ziyu Wang @ SparkCraft*

# A Visual Guide to MiniMax M2.7

*Architecture, training innovations, and the self-evolution breakthrough — explained with diagrams.*

**Week 4 — RL 101 Study Group**

---

MiniMax M2.7 is a 230-billion-parameter sparse Mixture-of-Experts model that
activates just **10 billion parameters per token** — making it one of the most
efficient frontier models available. Released in April 2026, it introduced a
remarkable property: an early checkpoint of the model acted as an autonomous ML
engineer during its own reinforcement learning cycle, achieving a **30% internal
performance lift** across 100+ rounds with no human intervention.

This guide walks through the key architectural and training innovations step by
step, with diagrams for each major concept.

> **Week 3 connection:** M2.7's RL training uses CISPO — a direct evolution of the
> PPO clipping we studied last week. Section 8 shows exactly what CISPO changes
> and why.

---

## Table of Contents

**Part I — Architecture (Sections 1-6)**
1. [Two Families, One Lab](#1-two-families-one-lab)
2. [Mixture of Experts: 256 Specialists, 8 at a Time](#2-mixture-of-experts-256-specialists-8-at-a-time)
3. [Why Sigmoid? The Routing Revolution](#3-why-sigmoid-the-routing-revolution)
4. [Grouped Query Attention (GQA)](#4-grouped-query-attention-gqa)
5. [Partial RoPE: Keeping Some Dimensions Position-Free](#5-partial-rope-keeping-some-dimensions-position-free)
6. [Multi-Token Prediction](#6-multi-token-prediction)

**Part II — From PPO to Production RL (Sections 7-12)**
7. [From PPO to Production RL](#7-from-ppo-to-production-rl) *-- bridges Weeks 1-3 to frontier RL*
8. [Forge: The RL Framework That Makes It All Possible](#8-forge-the-rl-framework-that-makes-it-all-possible)
9. [CISPO: Why Every Token Gets a Gradient](#9-cispo-why-every-token-gets-a-gradient)
10. [Reward Modeling: Three Signals for Agent Training](#10-reward-modeling-three-signals-for-agent-training)
11. [Prefix-Tree Merging: 40x Training Speedup](#11-prefix-tree-merging-40x-training-speedup)
12. [The Self-Evolution Loop](#12-the-self-evolution-loop)

**Part III — Results (Sections 13-16)**
13. [Agent Teams and Skill Adherence](#13-agent-teams-and-skill-adherence)
14. [Benchmark Performance](#14-benchmark-performance)
15. [Real-World Applications](#15-real-world-applications)
16. [Putting It All Together](#putting-it-all-together)

[Sources](#sources)

---

## 1. Two Families, One Lab

MiniMax maintains two entirely separate model architectures, and confusing them
leads to serious misunderstandings about M2.7's capabilities.

| | MiniMax-01 / M1 family | M2 series (M2 -> M2.7) |
|---|---|---|
| Total parameters | 456B | 230B |
| Active parameters | 45.9B (10%) | **10B (4.3%)** |
| Layers | 80 | 62 |
| Experts per layer | 32 | **256** |
| Attention | Hybrid: Lightning + softmax (7:1) | Pure softmax + GQA |
| Context window | 1 million tokens | 204,800 tokens |
| Expert routing | Softmax | **Sigmoid (novel)** |

> *The two MiniMax model families share no architecture — all differences between
> M2 versions (M2, M2.1, M2.5, M2.7) come from post-training improvements, not
> architectural changes.*

The M2 series made a deliberate choice: trade extreme context length (the 01
family's strength) for **radical inference efficiency**. With only 4.3% of
parameters active per token, M2.7 runs nearly as fast as a 10B dense model
despite carrying 230B of learned knowledge.

---

## 2. Mixture of Experts: 256 Specialists, 8 at a Time

Think of a dense language model as a single enormous restaurant where every chef
works on every order. A Mixture-of-Experts model is a food hall with 256
specialist kitchens. For each token, a routing system selects the 8 most relevant
kitchens to handle it. Only those 8 stoves are lit at any moment.

<!-- Diagram: MoE routing — input token → sigmoid router → 8 of 256 experts activated → weighted sum → output -->

```
Input token embedding
        |
  Sigmoid router
  (256 independent scores -> select top-8)
        |
  +-----+-----+-----+-----+-----+-----+-----+-----+
  |     |     |     |     |     |     |     |     |
 E-47 E-89 E-121 E-143 E-178 E-199 E-221 E-249
  [on] [on] [off] [off] [on]  [off] [on]  [on]
  +-----+-----+-----+-----+-----+-----+-----+-----+
        |
   weighted sum
        |
  Updated token embedding
```

Each token independently selects 8 of 256 experts per layer. The router uses
sigmoid scoring — each expert's score is independent of all others.

---

## 3. Why Sigmoid? The Routing Revolution

Most MoE models use **softmax routing**: expert scores are normalized so they sum
to 1.0. This creates competition — if Expert A's score rises, Expert B's must
fall. M2.7 uses **sigmoid routing** instead: each expert receives an independent
score between 0 and 1, with no constraint on the sum.

| | Softmax routing (most MoE) | Sigmoid routing (M2.7) |
|---|---|---|
| E-1 | 0.38 | 0.85 |
| E-2 | 0.30 | 0.72 |
| E-3 | 0.16 | 0.61 |
| E-4 | 0.09 | 0.38 |
| E-5 | 0.07 | 0.21 |
| **Sum** | **= 1.00** (scores compete) | **!= 1** (each is independent) |

With softmax, Expert 1 getting a higher score mathematically forces Expert 2's
score down. Sigmoid removes this coupling — each expert scores entirely on its
own merit. This leads to more stable training and better load balancing across 256
experts.

> **Industry convergence:** DeepSeek-V3 independently converged on sigmoid routing
> for the same reasons — further validating this design choice.

---

## 4. Grouped Query Attention (GQA)

M2.7 uses standard softmax self-attention across all 62 layers, but with a key
efficiency trick: **Grouped Query Attention**. Instead of each query head having
its own key-value pair, 6 query heads share a single KV head. With 48 query heads
and only 8 KV heads, the model stores **6x less in the KV cache** — critical for
long context.

```
One attention group (6 Q heads -> 1 KV head) x 8 groups

  Q head 1 ─┐
  Q head 2 ─┤
  Q head 3 ─┼──> KV head (shared by 6 Q, 128 dims)
  Q head 4 ─┤                     |
  Q head 5 ─┤               6x less KV
  Q head 6 ─┘               cache memory

Total: 48 Q heads + 8 KV heads · head dimension: 128
```

In standard multi-head attention, 48 query heads would require 48 KV heads. GQA
reduces this to just 8 KV heads — a 6x reduction in KV cache storage. NVIDIA
built a fused QK RMSNorm kernel specifically for M2.7 to speed this up further.

---

## 5. Partial RoPE: Keeping Some Dimensions Position-Free

Rotary Position Embeddings (RoPE) encode sequence position by rotating query and
key vectors. M2.7 uses a base frequency of **theta = 5,000,000** (500x the
original RoPE default) to support 200K tokens. But there's a subtler trick: RoPE
is applied to only *half* of each head's dimensions.

```
Each attention head: 128 dimensions

  Dim 0-63 (RoPE applied)          Dim 64-127 (no rotation)
  ┌─────────────────────┐          ┌─────────────────────┐
  │ positional info     │          │ semantic content    │
  │ rotates by position │          │ anchor — unchanged  │
  │ theta = 5,000,000   │          │ preserves meaning   │
  └─────────────────────┘          └─────────────────────┘

p-RoPE with p = 0.5 · enables stable long-context extrapolation
```

High-frequency RoPE pairs already carry sufficient positional information. The
low-frequency pairs, which are used for semantic content, benefit from no
rotation — they act as a stable anchor that doesn't shift with position. This
combination lets M2.7 handle sequences up to 204,800 tokens reliably.

---

## 6. Multi-Token Prediction

Standard language model training predicts one token at a time. M2.7 uses
**Multi-Token Prediction (MTP)**: three lightweight auxiliary modules
simultaneously predict tokens 2, 3, and 4 positions ahead. Each position now
generates four gradient signals instead of one, forcing the model to build
richer, forward-looking representations.

```
Input token sequence [ t1  t2  t3  ...  tn ]
                          |
            Main model (62 transformer layers)
            hidden states h1, h2, ... hn
                          |
        ┌─────────┬───────┼────────┬──────────┐
        |         |       |        |          |
    Main head   MTP-1   MTP-2   MTP-3
   predicts t+1  t+2     t+3     t+4

  4x richer gradient signal per training step
```

Each MTP module is a single lightweight transformer layer — minimal overhead, but
the extra gradient targets significantly improve sample efficiency and planning
ability. At inference, MTP enables **speculative decoding** for faster generation.

---

## 7. From PPO to Production RL

### What we've learned so far

Over the past three weeks, we built up the foundations of RL:

| Week | Topic | Key idea |
|---|---|---|
| 1 | MDPs and value functions | States, actions, rewards, discounting |
| 2 | DQN (value-based) | Learn $Q(s,a)$, act greedily, replay buffer, target network |
| 3 | PPO (policy-based) | Learn $\pi_\theta(a \mid s)$ + $V_\phi(s)$, clipped surrogate, GAE |

PPO gave us a stable way to train a policy with clipping. But our PPO ran on
**CartPole** — 4-dimensional observations, 2 actions, episodes of ~500 steps.

M2.7's RL training operates in a fundamentally different regime:

| | CartPole PPO (Week 3) | M2.7 Agent RL |
|---|---|---|
| Observation | 4 floats | 200,000 tokens of context |
| Action space | 2 discrete | ~100,000 token vocabulary |
| Episode length | ~500 steps | Hundreds of tool calls over minutes |
| Environments | 4 parallel | 100,000+ distinct scaffolds |
| Reward | +1 per step (dense) | Task completion (sparse) |
| Policy size | ~9K parameters | 230B parameters (10B active) |

The algorithm is recognizably PPO at its core — but every component must be
re-engineered for this scale.

### The LLM training pipeline

Modern frontier models go through multiple training stages. M2.7's pipeline:

```
Stage 1: Pre-training               Stage 2: Post-training
┌────────────────────┐               ┌──────────────────────────────┐
│  Massive text       │               │  SFT (supervised fine-tuning) │
│  corpus → predict   │    ──>        │  on curated demonstrations    │
│  next token         │               └──────────────┬───────────────┘
│  (230B params)      │                              │
└────────────────────┘               ┌──────────────v───────────────┐
                                     │  RLHF / RL from rewards      │
                                     │  (chat quality, safety)      │
                                     └──────────────┬───────────────┘
                                                    │
                                     ┌──────────────v───────────────┐
                                     │  Agent RL (Forge + CISPO)    │
                                     │  100K+ real-world envs       │
                                     │  tool use, code, multi-agent │
                                     └──────────────────────────────┘
```

**Pre-training** teaches the model language. **SFT** teaches it to follow
instructions. **RLHF** aligns it with human preferences. But **agent RL** — the
final stage — is where M2.7 learns to actually *do things*: write code, use
tools, coordinate with other agents, and recover from failures.

This is where our Week 3 PPO knowledge directly applies.

### Why agent RL is harder than chat RLHF

Standard RLHF (the kind that made ChatGPT work) trains on single-turn or
short-turn conversations. The reward comes from a trained reward model that
scores each response. Agent RL faces fundamentally harder challenges:

**Long horizons:** An agent debugging a codebase might take 50+ tool calls over
200K tokens of context. Credit assignment — figuring out *which action* caused
success or failure — becomes extremely difficult. This is why GAE (Week 3,
Section 3) matters: it propagates credit backwards through long trajectories.

**Sparse rewards:** In CartPole, every step gives +1 reward. In agent tasks, you
often get a single reward at the end: "did the code pass the tests?" Everything
before that is unrewarded. This makes the variance problem (Week 3, the reason
we need advantage estimation) much worse.

**Massive action space:** CartPole has 2 actions. An LLM has ~100K tokens it could
emit at each step. The policy gradient must work in this enormous space.

**Non-stationary environments:** Each of 100K+ task scaffolds is different. The
model can't memorize solutions — it must generalize.

**Off-policy data:** With 100K+ environments generating trajectories at different
speeds, the policy that generated a sample may have already been updated by the
time training happens. This is the off-policy problem that Forge's windowed FIFO
scheduling addresses.

MiniMax calls this the **"impossible triangle"**: you can't simultaneously have
high throughput, stable training, and flexible agent support with standard RL
infrastructure. Forge is their answer.

---

## 8. Forge: The RL Framework That Makes It All Possible

M2.7's most important training innovation is not architectural — it's the **Forge
reinforcement learning framework**. Forge resolves what MiniMax calls the
"impossible triangle" of training agent models: system throughput, training
stability, and support for arbitrary agent environments.

### The three-layer architecture

```
┌───────────────────────────────────────────────────────┐
│  Training / Inference Side                            │
│  ┌─────────────────┐    ┌──────────────────────────┐  │
│  │ Rollout engine   │    │ Training engine (CISPO)  │  │
│  └────────┬────────┘    └──────────────────────────┘  │
│           │                                           │
│  ┌────────┴────────┐    ┌──────────────────────────┐  │
│  │ Gateway server   │    │ Data pool (async)        │  │
│  └────────┬────────┘    └──────────────────────────┘  │
│           │                                           │
│  ┌────────┴───────────────────────────────────────┐   │
│  │ Agent side: agents + environments              │   │
│  │ (100,000+ distinct tasks, up to 200K context)  │   │
│  └────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
```

The three layers are fully decoupled. Agents can manipulate context in any way
(compress memory, run sub-agents, rewrite history) and Forge handles data
collection and gradient computation transparently.

### The 4-interface middleware

Agents integrate with Forge through exactly four interfaces:

| Interface | Purpose |
|---|---|
| **reprocess** | Transform/compress context before the next turn |
| **run** | Execute the agent's action in the environment |
| **postprocess** | Process the result (parse output, update memory) |
| **reward** | Compute the reward signal for the completed trajectory |

This design means any agent architecture — black-box tools, sub-agent
orchestration, arbitrary scaffolding — can plug into Forge without modification.
It enabled training across **over 100,000 distinct real-world environments**.

### Windowed FIFO scheduling

Forge uses a windowed FIFO strategy to balance data freshness against throughput.
Samples are consumed in approximate order of collection, with a window that
allows some out-of-order processing. This keeps the off-policy gap small (the
policy that generated the data is close to the current policy) while maintaining
high GPU utilization.

---

## 9. CISPO: Why Every Token Gets a Gradient

Forge uses a novel RL algorithm called **CISPO** (Clipped Importance Sampling
Policy Optimization). The problem it solves is subtle but important: in standard
PPO, tokens that change probability too quickly have their gradients **zeroed out
entirely**. This permanently prevents certain tokens — especially discourse
markers like "wait," "let me reconsider," "actually" — from being learned. CISPO
eliminates this dead zone.

> **Week 3 callback:** Recall PPO's clipped objective from our presentation:
>
> $$L^{\text{CLIP}} = \min\left(r \cdot A,\  \text{clip}(r, 1-\varepsilon, 1+\varepsilon) \cdot A\right)$$
>
> When $r$ (the probability ratio) leaves $[0.8, 1.2]$, PPO sets the gradient
> to **zero** through that token. CISPO changes *what* gets clipped.

### PPO vs CISPO: the key difference

| | PPO (ratio clip) | CISPO (weight clip) |
|---|---|---|
| "the" | normal gradient | full gradient |
| "wait" | **CLIPPED — zero gradient** | small but non-zero |
| "let" | normal gradient | good gradient |
| "rethink" | **CLIPPED — zero gradient** | small but non-zero |
| Tokens that learn | 2 of 4 | **All 4** |

**PPO clips the probability ratio** $r(\theta)$ — when it's outside the bounds, the
entire gradient for that token becomes zero.

**CISPO clips the importance sampling *weights*** (scalar multipliers on the loss)
rather than the ratio itself. Every token always receives a gradient — some
tokens just receive a smaller one.

This allows "discourse tokens" (wait, let me reconsider, actually...) to emerge
naturally through training. Experiments showed **2x faster convergence vs DAPO**
on Qwen2.5-32B.

### CISPO also adds early stopping

Beyond weight clipping, CISPO monitors for reward hacking — when the model finds
shortcuts that increase the reward signal without genuinely improving. When
detected, training stops early for that batch, preventing the model from
overfitting to reward artifacts.

---

## 10. Reward Modeling: Three Signals for Agent Training

In our CartPole PPO (Week 3), the reward was simple: +1 for every step the pole
stays up. For agent RL, designing the reward is one of the hardest problems.
MiniMax uses **three complementary reward signals**:

### Process rewards (intermediate feedback)

Rather than waiting until the end of a task to give a reward, process rewards
target **intermediate behaviors**. Examples:

- Penalizing language mixing (switching between Chinese and English mid-response)
- Penalizing specific tool invocation errors (wrong API format, missing arguments)
- Rewarding clean reasoning chains (well-structured step-by-step thinking)

This is similar to the "dense vs sparse reward" problem in RL: dense rewards
guide learning faster. Process rewards give the model feedback *during* a
trajectory, not just at the end.

### Task completion time reward

A novel signal: MiniMax incorporates **relative completion time** as a reward.
Faster task completion gets higher reward — but only relative to other attempts,
not absolute time. This incentivizes the model to:

- Parallelize tool calls when possible (e.g., run tests while editing another file)
- Avoid redundant steps
- Choose efficient strategies over thorough-but-slow ones

### Reward-to-go normalization

For long trajectories (200K tokens, dozens of tool calls), raw returns have
enormous variance — the same problem we saw with REINFORCE in Week 3 but far
worse. MiniMax normalizes **reward-to-go** (the return from each timestep
forward) to reduce gradient variance.

> **Week 3 callback:** This is conceptually identical to what GAE does — reducing
> variance in advantage estimation. The difference is scale: instead of 128 steps
> in CartPole, they normalize across trajectories with thousands of tokens.

### Unified multi-domain training

MiniMax trains across **three domains simultaneously**:

| Domain | Examples | Why it matters |
|---|---|---|
| **Reasoning** | Math, logic, planning | Builds the thinking backbone |
| **General QA** | Knowledge, conversation | Prevents forgetting |
| **Agent tasks** | Code, tools, multi-agent | The target capability |

Training these together rather than sequentially avoids **negative transfer** —
where improving one capability degrades another. The mixing strategy ensures the
model maintains general abilities while specializing in agent tasks.

---

## 11. Prefix-Tree Merging: 40x Training Speedup

Agent training generates many rollout samples that share a long common prefix —
the same system prompt, conversation history, and retrieved context. Naive
training computes these redundantly. Forge's **prefix-tree merging** identifies
shared prefixes and computes them exactly once.

```
Before: 3 separate passes               After: shared root + branches

A [shared prefix][task A]                       ┌─ [task A]
B [shared prefix][task B]     ──>  [shared   ]──┤
C [shared prefix][task C]           prefix    ]──┤
                                    [once only]  └─ [task C]
shared prefix computed 3x                shared prefix computed 1x
        (wasteful)                          (40x speedup)
```

In practice, long agent rollouts share enormous amounts of context. Prefix-tree
merging eliminates this redundancy at the sample level. The attention computation
remains mathematically identical — the merge is purely a computational
optimization.

This is what makes training across **100,000+ environments** economically viable.
Without it, the compute cost would be prohibitive.

---

## 12. The Self-Evolution Loop

The headline feature of M2.7 is that an early checkpoint operated as an
**autonomous ML engineer** during its own RL training — monitoring pipelines,
diagnosing failures, running optimization rounds, and committing code changes
with no human intervention. Over 100+ rounds, it achieved a **30% internal
performance gain**.

```
┌──────────────────────────────────────┐
│  M2.7 agent (early checkpoint)       │
│  short-term memory + self-feedback   │
└──────────────┬───────────────────────┘
               |
        ┌──────v──────────────────┐
        │ Analyze failure         │<─────────────┐
        │ trajectories            │              │
        └──────┬──────────────────┘              │
               |                                 │
        ┌──────v──────────────────┐              │
        │ Plan and commit         │        100+ rounds
        │ code changes            │        no human edits
        └──────┬──────────────────┘              │
               |                                 │
        ┌──────v──────────────────┐              │
        │ Run evaluation harness  │              │
        └──────┬──────────────────┘              │
               |                                 │
        ┌──────v──────────────────┐              │
        │ Compare vs baseline     │──────────────┘
        └─────────────────────────┘

Result: +30% performance on internal evaluation sets
```

The model discovered optimizations humans might miss:
- Systematically searching all files for the same bug pattern after a fix
- Adding loop detection to prevent infinite agent loops
- Finding optimal sampling parameter combinations

MiniMax reports M2.7 handles **30-50% of their RL team's daily engineering
workflow** through a "Research Agent Harness" that supports data pipelines,
training environments, infrastructure, cross-team collaboration, and persistent
memory.

---

## 13. Agent Teams and Skill Adherence

*This section covers capabilities from the official M2.7 release that go beyond
architecture into how the model operates as an agent system.*

### Native multi-agent collaboration

M2.7 supports **Agent Teams** — multi-agent collaboration requiring role
boundaries, adversarial reasoning, protocol adherence, and behavioral
differentiation. MiniMax emphasizes these cannot be achieved through prompting
alone; they require RL training on multi-agent scenarios.

### Skill adherence at scale

One of the most impressive numbers in the release: **97% skill compliance rate
across 40 complex skills**, each exceeding 2,000 tokens in specification length.
This means M2.7 can reliably follow detailed, multi-page instruction sets —
critical for enterprise deployment where tools have precise protocols.

### The 4 pillars of agent capability

MiniMax frames M2.7's agent abilities around four pillars:

| Pillar | What it means | Key metric |
|---|---|---|
| **Coding** | End-to-end software engineering | SWE-Pro 56.2% |
| **Tools** | Reliable interaction with external APIs | Toolathon 46.3% |
| **Skills** | Following complex multi-page protocols | 97% compliance |
| **Teams** | Multi-agent coordination | Native support |

---

## 14. Benchmark Performance

### Headline numbers

| Metric | Score |
|---|---|
| SWE-Pro | **56.2%** (matches GPT-5.3 Codex) |
| SWE-bench Verified | **~78%** |
| Hallucination rate | **34%** (lowest tier) |
| API pricing | **$0.30 / 1M input tokens** |

### Full benchmark table

*Scores from MiniMax's official M2.7 release (April 2026):*

| Benchmark | M2.7 Score | Context |
|---|---|---|
| SWE-Pro | 56.22% | Software engineering; matches GPT-5.3 Codex |
| VIBE-Pro | 55.6% | End-to-end project delivery |
| Terminal Bench 2 | 57.0% | Complex engineering systems |
| SWE Multilingual | 76.5 | Real-world multilingual scenarios |
| Multi SWE Bench | 52.7 | Multi-domain programming |
| GDPval-AA ELO | 1495 | Professional domain expertise (highest open-source) |
| Toolathon | 46.3% | Tool interaction |
| MM Claw | 62.7% | Professional work tasks (comparable to Sonnet 4.6) |
| NL2Repo | 39.8% | System comprehension |
| MLE Bench Lite | 66.6% avg | ML competitions (9 gold, 5 silver, 1 bronze) |

**Competitive context:** on MLE Bench Lite, M2.7 ranks second only to Opus 4.6
(75.7%) and GPT-5.4 (71.2%), tying with Gemini 3.1.

### Cost efficiency

M2.7 achieves near-frontier performance at roughly **1/10th of frontier pricing**.
The 4.3% activation ratio (10B of 230B) means inference costs scale with the
active parameter count, not the total — the same economic advantage that makes
MoE architectures attractive for deployment.

---

## 15. Real-World Applications

The official M2.7 release highlights several deployed capabilities:

**Live debugging:** Reduced recovery time for production system incidents to
**under three minutes** — the model diagnoses issues, identifies root causes, and
proposes fixes in real-time.

**Office tools:** Enhanced Excel, PowerPoint, and Word editing with multi-round
revisions and high-fidelity editing. The model handles complex formatting and
cross-reference updates.

**Financial modeling:** TSMC revenue forecasting producing "first draft" quality
outputs — the model structures financial models, pulls relevant data, and
generates projections.

**Entertainment:** "OpenRoom" demo framework enabling interactive character
scenarios for game and creative applications.

---

## Putting It All Together

What makes M2.7 coherent as a system is that every design choice reinforces a
single goal: *maximum agentic capability per unit of compute*.

| Innovation | Why it matters |
|---|---|
| Sigmoid routing + 256 experts | Stable training, 10B active params = cheap inference |
| GQA at 6:1 ratio | Long context affordable (6x KV cache reduction) |
| Partial RoPE at theta=5M | 200K context without compromising reasoning |
| CISPO (not PPO) | Discourse tokens emerge; 2x faster convergence |
| Prefix-tree merging | 40x speedup makes 100K+ env training viable |
| Self-evolution loop | Model improves its own training; 30% gain |
| Agent Teams + Skills | Multi-agent coordination + 97% skill compliance |

**Three insights from M2.7 likely to spread across the field:**

1. **Sigmoid routing over softmax** for large-scale MoE — DeepSeek-V3
   independently converged on the same choice
2. **Prefix-tree merging** as the critical efficiency technique for agent RL
   training
3. **Models can meaningfully participate in their own training pipelines** — not
   as architects, but as tireless operational engineers

M2.7 is a text-only model with a 200K context window and a proprietary license.
For teams building agentic software engineering pipelines, it offers a compelling
argument: near-frontier performance at roughly a tenth of frontier pricing.

---

## Sources

- MiniMax, [*MiniMax M2.7*](https://www.minimax.io/news/minimax-m27-en) (April 2026) — official release blog
- MiniMax, [*Forge: Scalable Agent RL Framework and Algorithm*](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) (April 2026) — Forge framework and CISPO details
- MiniMax, [*MiniMax M2.5*](https://www.minimax.io/news/minimax-m25) (April 2026) — M2.5 release (predecessor context)
- Turing Post, [*Inside a Chinese AI Lab: How MiniMax Uses RL*](https://turingpost.substack.com/p/inside-a-chinese-ai-lab-how-minimax) (February 2026) — interview on RL debugging
- Grootendorst, [*A Visual Guide to Gemma 4*](https://open.substack.com/pub/maartengrootendorst/p/a-visual-guide-to-gemma-4) — style reference for this guide

---

*RL 101 Study Group — Week 4*
*Colby Ziyu Wang @ SparkCraft*

# Week 4: RLHF and the Path to Agent RL

*From human preferences to autonomous ML engineers — the full post-training
stack, explained with a MiniMax M2.7 case study.*

**Week 4 — RL 101 Study Group**

---

In Weeks 1-3 we built up the foundations: MDPs, value functions, DQN, and PPO.
This week we zoom out to the **post-training pipeline** that turns a raw
language model into a useful, aligned, and capable agent. We cover RLHF
(Reinforcement Learning from Human Feedback) end to end, then see how MiniMax
applies these ideas at scale in M2.7.

> **Reference text:** Nathan Lambert,
> [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) (2025).
> This guide draws on Lambert's book structure — we recommend it for deeper
> study.

---

## Table of Contents

### Part I — RLHF Foundations

1. [What is RLHF?](#1-what-is-rlhf)
2. [The Training Pipeline](#2-the-training-pipeline)
3. [Instruction Fine-Tuning (SFT)](#3-instruction-fine-tuning-sft)
4. [Reward Models: Learning What Humans Want](#4-reward-models-learning-what-humans-want)
5. [Policy Optimization: From REINFORCE to PPO](#5-policy-optimization-from-reinforce-to-ppo)
6. [Direct Alignment: DPO and the RL-Free Alternative](#6-direct-alignment-dpo-and-the-rl-free-alternative)
7. [Reasoning with RL](#7-reasoning-with-rl)
8. [The Challenges: Reward Hacking and Over-Optimization](#8-the-challenges-reward-hacking-and-over-optimization)
9. [From RLHF to Agent RL](#9-from-rlhf-to-agent-rl)

### Part II — Case Study: MiniMax M2.7 Architecture

10. [Two Families, One Lab](#10-two-families-one-lab)
11. [Mixture of Experts: 256 Specialists, 8 at a Time](#11-mixture-of-experts-256-specialists-8-at-a-time)
12. [Why Sigmoid? The Routing Revolution](#12-why-sigmoid-the-routing-revolution)
13. [Grouped Query Attention (GQA)](#13-grouped-query-attention-gqa)
14. [Partial RoPE: Keeping Some Dimensions Position-Free](#14-partial-rope-keeping-some-dimensions-position-free)
15. [Multi-Token Prediction](#15-multi-token-prediction)

### Part III — Case Study: MiniMax M2.7 RL at Scale

16. [Forge: The RL Framework](#16-forge-the-rl-framework)
17. [CISPO: Why Every Token Gets a Gradient](#17-cispo-why-every-token-gets-a-gradient)
18. [MiniMax Reward Modeling: Three Signals](#18-minimax-reward-modeling-three-signals)
19. [Prefix-Tree Merging: 40x Training Speedup](#19-prefix-tree-merging-40x-training-speedup)
20. [The Self-Evolution Loop](#20-the-self-evolution-loop)

### Part IV — Results

21. [Agent Teams and Skill Adherence](#21-agent-teams-and-skill-adherence)
22. [Benchmark Performance](#22-benchmark-performance)
23. [Real-World Applications](#23-real-world-applications)
24. [Putting It All Together](#24-putting-it-all-together)

[Sources](#sources)

---

# Part I — RLHF Foundations

---

## 1. What is RLHF?

**Reinforcement Learning from Human Feedback** is the process of training an AI
model to behave the way humans want, using human preferences as the reward
signal.

The core insight: it's often easier for humans to **compare** two outputs than to
**write** the ideal output. RLHF exploits this by:

1. Showing humans pairs of model outputs
2. Training a **reward model** on those preferences
3. Using RL (typically PPO) to optimize the language model against that reward

```
                  Human annotators
                  "A is better than B"
                        |
                        v
              ┌─────────────────────┐
              │   Reward Model      │
              │   R(prompt, response)│
              │   = scalar score    │
              └──────────┬──────────┘
                         |
                         v
              ┌─────────────────────┐
              │   PPO / RL          │
              │   optimize policy   │
              │   to maximize R     │
              └─────────────────────┘
```

**Why not just do supervised learning?** Because we don't have the "right answer"
for most prompts. There's no ground truth for "write a poem about autumn" — but
a human can easily say which of two poems they prefer. RLHF converts these
relative judgments into a training signal.

**Historical context:** RLHF wasn't invented for LLMs. The idea of learning from
human comparisons dates back to economics (utility theory) and optimal control.
Christiano et al. (2017) applied it to Atari games. OpenAI's InstructGPT (2022)
and then ChatGPT (2022) brought it to language models — and changed everything.

---

## 2. The Training Pipeline

Modern frontier models go through a multi-stage pipeline. Each stage builds on
the previous:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Stage 1: PRE-TRAINING                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Predict next token on trillions of tokens             │  │
│  │  Model learns language, facts, reasoning patterns      │  │
│  │  Output: a capable but unaligned "base model"          │  │
│  └────────────────────────────────────────────────────────┘  │
│                           |                                  │
│  Stage 2: SUPERVISED FINE-TUNING (SFT)                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Train on curated (prompt, ideal_response) pairs       │  │
│  │  Model learns the format of being an assistant         │  │
│  │  Output: follows instructions, but quality varies      │  │
│  └────────────────────────────────────────────────────────┘  │
│                           |                                  │
│  Stage 3: REWARD MODEL TRAINING                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Collect human preferences (A > B for each prompt)     │  │
│  │  Train a reward model to predict human preferences     │  │
│  │  Output: R(prompt, response) = scalar quality score    │  │
│  └────────────────────────────────────────────────────────┘  │
│                           |                                  │
│  Stage 4: RL OPTIMIZATION (PPO / RLHF)                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Generate responses, score with reward model           │  │
│  │  Update policy with PPO to maximize reward             │  │
│  │  KL penalty keeps model close to SFT baseline          │  │
│  │  Output: aligned, high-quality model                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  (Optional) Stage 5: AGENT RL                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Train on tool use, code, multi-step tasks             │  │
│  │  Reward = task success (not human preference)          │  │
│  │  Output: model that can act in the real world          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Canonical recipes:**

| Recipe | Stages | Notable for |
|---|---|---|
| **InstructGPT** (OpenAI, 2022) | SFT -> RM -> PPO | First large-scale RLHF |
| **Tulu 3** (AI2, 2024) | SFT -> DPO + PPO | Open-source, reproducible |
| **DeepSeek R1** (2025) | SFT -> RL (reasoning) | RL for chain-of-thought |
| **MiniMax M2.7** (2026) | SFT -> RLHF -> Agent RL (Forge) | Self-evolving agent training |

> **Week 3 connection:** Stage 4 is literally PPO — the same algorithm we
> implemented for CartPole. The policy is the LLM, the action is the next token,
> and the reward comes from the reward model instead of the environment.

---

## 3. Instruction Fine-Tuning (SFT)

The base model from pre-training is a powerful next-token predictor, but it
doesn't know it should *answer questions* or *follow instructions*. SFT teaches
it the format.

**Training data:** Curated pairs of (instruction, ideal response). Examples:

| Instruction | Response |
|---|---|
| "Summarize this article in 3 bullet points" | "- Point 1\n- Point 2\n- Point 3" |
| "Write a Python function that sorts a list" | "```python\ndef sort_list(...)..." |
| "Explain quantum entanglement simply" | "Imagine two coins that always..." |

**Training objective:** Standard cross-entropy loss — exactly like pre-training,
but only on the response tokens (the instruction tokens are masked).

**Key insight from Lambert (Ch. 4):** SFT quality matters enormously. A small
dataset of high-quality demonstrations (1K-10K examples) often outperforms a
large dataset of mediocre ones. The Alpaca/Vicuna era showed that even simple
SFT on GPT-4 outputs could dramatically improve open-source models.

**Limitations:** SFT teaches the model to *imitate* demonstrations. It can't
learn to be *better* than the demonstrations. For that, you need RL.

---

## 4. Reward Models: Learning What Humans Want

The reward model (RM) is the bridge between human judgment and RL. It takes a
(prompt, response) pair and outputs a scalar score representing quality.

### The Bradley-Terry preference model

Given two responses $y_w$ (preferred / "winner") and $y_l$ (rejected / "loser")
to the same prompt $x$, the **Bradley-Terry model** says the probability that
the human prefers $y_w$ is:

$$
P(y_w \succ y_l \mid x) = \sigma\left(R(x, y_w) - R(x, y_l)\right)
$$

where $\sigma$ is the sigmoid function and $R$ is the reward model.

The RM is trained by minimizing the **negative log-likelihood** of the observed
preferences:

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}\left[\log \sigma\left(R(x, y_w) - R(x, y_l)\right)\right]
$$

This is called the **preference margin loss**. The RM learns to assign higher
scores to preferred responses. Note: only the *difference* in scores matters,
not the absolute values.

> **Week 2 callback:** This is conceptually similar to the TD loss in DQN — both
> train a neural network to predict a scalar value. The difference is the
> training signal: DQN uses Bellman targets, the RM uses human preferences.

### Outcome vs Process Reward Models

| | Outcome RM (ORM) | Process RM (PRM) |
|---|---|---|
| Scores | The entire response | Each reasoning step |
| Signal | Sparse (one score per response) | Dense (score per step) |
| Use case | General chat quality | Math/reasoning chains |
| Analogy | "Did you solve the problem?" | "Is each step correct?" |

**Process Reward Models** (Lightman et al., 2023) are particularly relevant for
reasoning: they score each intermediate step, providing denser feedback. MiniMax
uses a variant of process rewards in their agent training (Section 18).

### Practical considerations

- **Data collection:** Preference data is expensive. Typical datasets have
  10K-100K preference pairs. Constitutional AI (Anthropic, 2022) showed you can
  generate *synthetic* preferences using the model itself.
- **K-wise comparisons:** Instead of pairs, you can rank K responses. This
  extracts more information per annotation.
- **Reward model size:** The RM is usually a copy of the SFT model with the
  language modeling head replaced by a scalar output head.

---

## 5. Policy Optimization: From REINFORCE to PPO

This is where Weeks 1-3 directly connect to RLHF. The language model is the
**policy**, each generated token is an **action**, and the reward model provides
the **reward**.

### The RLHF objective

The goal is to maximize the expected reward while staying close to the SFT
model (to prevent the policy from degenerating):

$$
\max_\theta\ \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(\cdot \mid x)}\left[R(x, y)\right] - \beta \cdot \text{KL}\left[\pi_\theta \| \pi_{\text{SFT}}\right]
$$

The first term maximizes reward. The second term (KL penalty, weighted by
$\beta$) prevents the model from straying too far from the SFT baseline.
Without the KL term, the model would find degenerate outputs that "hack" the
reward model.

### PPO in the RLHF loop

The standard RLHF loop uses PPO (Week 3) with these components mapped:

| PPO concept (Week 3) | RLHF equivalent |
|---|---|
| Environment | Prompt dataset + reward model |
| State $s$ | Prompt + tokens generated so far |
| Action $a$ | Next token from vocabulary |
| Policy $\pi_\theta$ | The language model |
| Reward $r$ | $R(x, y) - \beta \cdot \text{KL}$ |
| Value function $V_\phi$ | Critic head (estimates expected reward) |
| Advantage (GAE) | Same — propagates credit across tokens |
| Clipped surrogate | Same — prevents destructive updates |

The per-token KL penalty is typically:

$$
\text{KL}_t = \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{SFT}}(a_t \mid s_t)}
$$

This is subtracted from the reward at each token, creating a per-token "cost"
for deviating from the SFT policy.

### Alternatives to PPO

The RLHF field has developed several alternatives:

| Algorithm | Key idea | Trade-off |
|---|---|---|
| **PPO** | Clipped surrogate + critic | Stable but complex (needs value network) |
| **REINFORCE** / **RLOO** | No critic, use leave-one-out baseline | Simpler but higher variance |
| **GRPO** | Group relative policy optimization | No critic, normalizes within group |
| **DPO** | Skip RL entirely (see Section 6) | Simple but less flexible |
| **CISPO** (MiniMax) | Weight clipping, not ratio clipping | Better for long-horizon agent tasks |

> **RLOO** (REINFORCE Leave-One-Out): generates K responses per prompt, uses the
> average reward of the other K-1 as a baseline. Simpler than PPO (no critic
> network) but requires multiple samples per prompt.
>
> **GRPO** (Group Relative Policy Optimization, DeepSeek): similar to RLOO but
> normalizes advantages within each group. Used in DeepSeek R1's reasoning
> training.

---

## 6. Direct Alignment: DPO and the RL-Free Alternative

**Direct Preference Optimization** (Rafailov et al., 2023) showed you can skip
the reward model and RL loop entirely.

### The key insight

The RLHF objective (maximize reward with KL penalty) has a **closed-form
solution**. DPO derives the optimal policy directly:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

This looks complex, but the intuition is simple: **increase the probability of
preferred responses and decrease the probability of rejected responses**, scaled
by how much the current policy differs from the reference.

### DPO vs PPO

| | PPO (RL-based) | DPO (direct) |
|---|---|---|
| Needs reward model? | Yes | No |
| Needs RL loop? | Yes (generate, score, update) | No (single supervised pass) |
| Training complexity | High (PPO + critic + RM) | Low (one loss function) |
| Compute | Expensive (generation + training) | Cheaper |
| Flexibility | Can optimize any reward | Only preference pairs |
| Quality ceiling | Higher (iterative improvement) | Lower (one-shot) |

**In practice:** Many teams use both. DPO for an initial alignment pass (cheap,
fast), then PPO for final polishing (expensive, better). Tulu 3 demonstrated
this hybrid approach.

---

## 7. Reasoning with RL

A recent breakthrough: using RL not just for alignment (being helpful/harmless)
but for **making the model smarter**.

### The o1 / R1 paradigm

OpenAI's o1 (2024) and DeepSeek's R1 (2025) showed that RL can teach models
to produce **chain-of-thought reasoning** — long internal monologues that work
through problems step by step.

The key ingredients:

1. **Outcome reward:** check if the final answer is correct (math, code tests)
2. **Process reward (optional):** score each reasoning step
3. **Long generation:** allow the model to "think" for hundreds or thousands of
   tokens before answering
4. **RL training:** PPO or GRPO optimizes for correct final answers

The model learns to produce reasoning patterns that lead to correct answers —
even patterns that weren't in the training data. This is fundamentally different
from SFT, which can only imitate existing reasoning.

**Why RL works for reasoning now (Lambert, Ch. 7):**
- Verifiable rewards (math has ground truth, code has tests)
- Sufficient base model capability (needs strong pre-training)
- Enough compute for long-horizon RL

### The reward is the key

| Task type | Reward signal | Verifiable? |
|---|---|---|
| Math | Correct final answer | Yes |
| Code | Tests pass | Yes |
| General chat | Reward model score | No (proxy) |
| Agent tasks | Task completion | Mostly yes |

RL for reasoning works best when rewards are **verifiable** — you can
automatically check correctness. This is why math and code improved first.

---

## 8. The Challenges: Reward Hacking and Over-Optimization

RLHF is not a free lunch. Several failure modes can derail training.

### Reward hacking

The model finds outputs that score high on the reward model **without actually
being good**. Examples:

- Responses that are verbose and confident-sounding but wrong
- Excessive hedging ("As an AI language model, I...")
- Exploiting quirks in the reward model's training data

**The Goodhart's Law problem:** "When a measure becomes a target, it ceases to be
a good measure." The reward model is a proxy for human judgment — optimizing it
too aggressively finds the gaps between the proxy and reality.

### KL regularization

The main defense: the KL penalty term $\beta \cdot \text{KL}[\pi_\theta \| \pi_{\text{SFT}}]$
keeps the policy close to the SFT baseline. Too much KL penalty and the model
barely changes. Too little and it reward-hacks.

**Practical tuning:** $\beta$ is one of the most important hyperparameters in
RLHF. Lambert (Ch. 15) covers implicit regularization, margin-based
regularization, and the interplay between explicit KL and implicit constraints.

### Over-optimization

Gao et al. (2022) showed that reward model score initially increases with
training, then **decreases** after a critical point — even though the RM score
keeps going up. The model is optimizing the reward model's blind spots.

```
Actual quality  ^
                |        /\
                |       /  \
                |      /    \____
                |     /
                |    /
                |   /
                └──────────────────> RL training steps
                    sweet    over-
                    spot     optimized
```

**Defenses:** early stopping, ensembling multiple reward models, rejection
sampling (Section 9 in Lambert's book), and process reward models that provide
denser, harder-to-hack signals.

---

## 9. From RLHF to Agent RL

Standard RLHF trains on single-turn conversations: one prompt, one response,
one reward. **Agent RL** extends this to multi-step interactions where the model
uses tools, writes code, and takes actions in the real world.

### What changes

| | Chat RLHF | Agent RL |
|---|---|---|
| Episode | Single turn | Dozens of tool calls |
| Context | Short (1-2K tokens) | Long (up to 200K tokens) |
| Reward | Preference model score | Task completion |
| Actions | Generate text | Generate text + call tools |
| Horizon | Short | Very long |
| Credit assignment | Easy (one response) | Hard (which action mattered?) |

### Why this matters

Agent RL is where the field is heading. Every major lab is investing in models
that can:
- Write and debug code autonomously
- Use web browsers and APIs
- Coordinate as multi-agent teams
- Run experiments and iterate

The rest of this guide examines **MiniMax M2.7** — one of the most detailed
public case studies of agent RL at scale. Their Forge framework, CISPO
algorithm, and self-evolution loop show what production agent RL looks like when
you scale PPO to 100,000+ real-world environments.

---

# Part II — Case Study: MiniMax M2.7 Architecture

MiniMax M2.7 is a 230-billion-parameter sparse Mixture-of-Experts model that
activates just **10 billion parameters per token**. Released in April 2026, an
early checkpoint acted as an autonomous ML engineer during its own RL training,
achieving a **30% performance lift** across 100+ rounds with no human
intervention.

---

## 10. Two Families, One Lab

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

## 11. Mixture of Experts: 256 Specialists, 8 at a Time

Think of a dense language model as a single enormous restaurant where every chef
works on every order. A Mixture-of-Experts model is a food hall with 256
specialist kitchens. For each token, a routing system selects the 8 most relevant
kitchens to handle it. Only those 8 stoves are lit at any moment.

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

## 12. Why Sigmoid? The Routing Revolution

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

## 13. Grouped Query Attention (GQA)

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

## 14. Partial RoPE: Keeping Some Dimensions Position-Free

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

## 15. Multi-Token Prediction

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

# Part III — Case Study: MiniMax M2.7 RL at Scale

This is where MiniMax applies the RLHF concepts from Part I at production
scale. Every component from the standard pipeline (reward models, PPO, KL
regularization) reappears here — re-engineered for 100,000+ environments and
200K-token contexts.

| RLHF concept (Part I) | MiniMax's version |
|---|---|
| PPO clipping | CISPO — weight clipping, not ratio clipping |
| Reward model | Process rewards + completion time + reward-to-go |
| KL regularization | Built into CISPO + early stopping |
| Training loop | Forge — decoupled 3-layer architecture |
| Preference data | 100K+ real-world task environments |

---

## 16. Forge: The RL Framework

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
Samples are consumed in approximate order of collection, with a window of size
$W$ (e.g., $W = 0.3N$ where $N$ is batch size). Within the window, any completed
trajectory can be fetched (greedy). Outside the window, fetching is forbidden —
preventing distribution shift toward fast, easy samples. This keeps the off-policy
gap small while maintaining high GPU utilization.

---

## 17. CISPO: Why Every Token Gets a Gradient

Forge uses a novel RL algorithm called **CISPO** (Clipped Importance Sampling
Policy Optimization). The problem it solves is subtle but important: in standard
PPO, tokens that change probability too quickly have their gradients **zeroed out
entirely**. This permanently prevents certain tokens — especially discourse
markers like "wait," "let me reconsider," "actually" — from being learned. CISPO
eliminates this dead zone.

> **Part I callback:** Recall PPO's clipped objective (Section 5):
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

Beyond weight clipping, CISPO monitors for **reward hacking** (Section 8) — when
the model finds shortcuts that increase the reward signal without genuinely
improving. When detected, training stops early for that batch, preventing the
model from overfitting to reward artifacts.

---

## 18. MiniMax Reward Modeling: Three Signals

In CartPole PPO (Week 3), the reward was simple: +1 for every step the pole
stays up. For agent RL, designing the reward is one of the hardest problems
(Section 8 — Goodhart's Law). MiniMax uses **three complementary reward signals**:

### Process rewards (intermediate feedback)

Rather than waiting until the end of a task to give a reward, process rewards
target **intermediate behaviors** (see ORM vs PRM in Section 4). Examples:

- Penalizing language mixing (switching between Chinese and English mid-response)
- Penalizing specific tool invocation errors (wrong API format, missing arguments)
- Rewarding clean reasoning chains (well-structured step-by-step thinking)

This is the "dense vs sparse reward" problem: dense rewards guide learning
faster. Process rewards give the model feedback *during* a trajectory, not just
at the end.

### Task completion time reward

A novel signal: MiniMax incorporates **relative completion time** as a reward.
Faster task completion gets higher reward — but only relative to other attempts,
not absolute time. This incentivizes the model to:

- Parallelize tool calls when possible
- Avoid redundant steps
- Choose efficient strategies over thorough-but-slow ones

### Reward-to-go normalization

For long trajectories (200K tokens, dozens of tool calls), raw returns have
enormous variance — the same problem we saw with REINFORCE in Week 3 but far
worse. MiniMax normalizes **reward-to-go** (the return from each timestep
forward) to reduce gradient variance.

### Unified multi-domain training

MiniMax trains across **three domains simultaneously**:

| Domain | Examples | Why it matters |
|---|---|---|
| **Reasoning** | Math, logic, planning | Builds the thinking backbone |
| **General QA** | Knowledge, conversation | Prevents forgetting |
| **Agent tasks** | Code, tools, multi-agent | The target capability |

Training these together rather than sequentially avoids **negative transfer** —
where improving one capability degrades another.

---

## 19. Prefix-Tree Merging: 40x Training Speedup

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

---

## 20. The Self-Evolution Loop

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

# Part IV — Results

---

## 21. Agent Teams and Skill Adherence

### Native multi-agent collaboration

M2.7 supports **Agent Teams** — multi-agent collaboration requiring role
boundaries, adversarial reasoning, protocol adherence, and behavioral
differentiation. MiniMax emphasizes these cannot be achieved through prompting
alone; they require RL training on multi-agent scenarios.

### Skill adherence at scale

**97% skill compliance rate across 40 complex skills**, each exceeding 2,000
tokens in specification length. This means M2.7 can reliably follow detailed,
multi-page instruction sets — critical for enterprise deployment where tools have
precise protocols.

### The 4 pillars of agent capability

| Pillar | What it means | Key metric |
|---|---|---|
| **Coding** | End-to-end software engineering | SWE-Pro 56.2% |
| **Tools** | Reliable interaction with external APIs | Toolathon 46.3% |
| **Skills** | Following complex multi-page protocols | 97% compliance |
| **Teams** | Multi-agent coordination | Native support |

---

## 22. Benchmark Performance

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

## 23. Real-World Applications

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

## 24. Putting It All Together

### The RLHF pipeline in practice

This guide traced the full arc from theory to production:

| Part | What we covered | Key takeaway |
|---|---|---|
| **Part I** | RLHF foundations | Reward models + PPO = aligned models |
| **Part II** | M2.7 architecture | MoE + sigmoid routing + GQA = cheap inference |
| **Part III** | M2.7 RL at scale | Forge + CISPO + prefix-tree = agent training |
| **Part IV** | Results | Near-frontier at 1/10th the cost |

### Three insights from M2.7 likely to spread

1. **Sigmoid routing over softmax** for large-scale MoE — DeepSeek-V3
   independently converged on the same choice
2. **Prefix-tree merging** as the critical efficiency technique for agent RL
   training
3. **Models can meaningfully participate in their own training pipelines** — not
   as architects, but as tireless operational engineers

### What makes M2.7 coherent

Every design choice reinforces a single goal: *maximum agentic capability per
unit of compute*.

| Innovation | Why it matters |
|---|---|
| Sigmoid routing + 256 experts | Stable training, 10B active params = cheap inference |
| GQA at 6:1 ratio | Long context affordable (6x KV cache reduction) |
| Partial RoPE at theta=5M | 200K context without compromising reasoning |
| CISPO (not PPO) | Discourse tokens emerge; 2x faster convergence |
| Prefix-tree merging | 40x speedup makes 100K+ env training viable |
| Self-evolution loop | Model improves its own training; 30% gain |
| Agent Teams + Skills | Multi-agent coordination + 97% skill compliance |

---

## Sources

**Books and courses:**
- Lambert, [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) (2025) — comprehensive RLHF textbook covering the full pipeline

**Papers:**
- Christiano et al., [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) (2017) — original RLHF
- Ouyang et al., [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) (DPO, 2023)
- Lightman et al., [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050) (Process Reward Models, 2023)
- Gao et al., [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) (2022)

**MiniMax sources:**
- MiniMax, [*MiniMax M2.7*](https://www.minimax.io/news/minimax-m27-en) (April 2026) — official release blog
- MiniMax, [*Forge: Scalable Agent RL Framework and Algorithm*](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) (April 2026) — Forge framework and CISPO
- Turing Post, [*Inside a Chinese AI Lab: How MiniMax Uses RL*](https://turingpost.substack.com/p/inside-a-chinese-ai-lab-how-minimax) (February 2026)

**Style reference:**
- Grootendorst, [*A Visual Guide to Gemma 4*](https://open.substack.com/pub/maartengrootendorst/p/a-visual-guide-to-gemma-4)

---

*RL 101 Study Group — Week 4*
*Colby Ziyu Wang @ SparkCraft*

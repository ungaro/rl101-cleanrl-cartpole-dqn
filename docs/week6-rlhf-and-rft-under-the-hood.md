---
marp: true
theme: default
paginate: true
header: "RL 101 — Week 6 — RLHF & Reinforcement Fine-Tuning Under the Hood"
footer: "rl101-cleanrl-cartpole-dqn"
math: katex
---

<!--
To view as slides, install marp-cli and run:
  npx @marp-team/marp-cli docs/week6-rlhf-and-rft-under-the-hood.md --preview
  npx @marp-team/marp-cli docs/week6-rlhf-and-rft-under-the-hood.md --pdf
On GitHub this file also renders as a normal document with math and code.
-->

# RLHF & Reinforcement Fine-Tuning Under the Hood

## Companion to RL 105 — Reinforcement Fine-Tuning and RLHF

**Week 6 — RL 101 Study Group**

A slide-by-slide deep dive into Colby's RL 105 lecture, with a research
appendix on the post-DPO landscape (KTO, ORPO, SimPO, IPO), the post-PPO
landscape (GRPO, DAPO, CISPO), and the RLVR shift that's quietly eating
classical RLHF for reasoning tasks.

---

## How this doc relates to other weeks

- **Week 2 (DQN)** built up Q-learning and the Bellman view.
- **Week 3 (PPO)** introduced the actor-critic split and the clipped surrogate.
- **Week 4 (Forge)** showed agent-RL systems and the **PPO → GRPO → DAPO → CISPO** lineage.
- **Week 5 (M2.7)** walked the full RLHF stack with a production case study.
- **Week 6 (this doc)** is the canonical *lecture* companion. It re-derives the math the slides condense, then surveys what's changed since the slides were written.

If you want depth, read Week 5 for the systems view and Week 4 for the
RL-algorithm view. This doc is the math + literature spine.

---

## Table of Contents

### Part I — Slide-by-Slide Companion (RL 105)

1. [LLMs as Policies](#1-llms-as-policies-slide-2)
2. [Why Fine-Tune?](#2-why-fine-tune-slide-3)
3. [Three Stages of LLM Training](#3-three-stages-of-llm-training-slide-4)
4. [What is Reinforcement Fine-Tuning?](#4-what-is-reinforcement-fine-tuning-slide-5)
5. [Three Levels of Abstraction](#5-three-levels-of-abstraction-slide-6)
6. [RLHF in One Sentence](#6-rlhf-in-one-sentence-slide-7)
7. [The RLHF Pipeline](#7-the-rlhf-pipeline-slide-8)
8. [Reward Models](#8-reward-models-slide-9)
9. [Preference Modeling — Where the Loss Comes From](#9-preference-modeling-slide-10)
10. [The LM as a Policy](#10-the-lm-as-a-policy-slide-11)
11. [The RLHF Objective Derived](#11-the-rlhf-objective-derived-slide-12)
12. [Why the KL Penalty?](#12-why-the-kl-penalty-slide-13)
13. [PPO for RLHF](#13-ppo-for-rlhf-slide-14)
14. [RLHF as Actor-Critic](#14-rlhf-as-actor-critic-slide-15)
15. [The RLHF Loop in Code](#15-the-rlhf-loop-in-code-slide-16)
16. [RFT vs RLHF vs SFT](#16-rft-vs-rlhf-vs-sft-slide-17)
17. [Why RLHF Matters](#17-why-rlhf-matters-slide-18)
18. [Limitations](#18-limitations-slide-19)

### Part II — What's New Since the Slides

19. [DPO and the RL-Free Family](#19-dpo-and-the-rl-free-family)
20. [Post-PPO: GRPO, DAPO, CISPO](#20-post-ppo-grpo-dapo-cispo)
21. [RLVR — Verifiable Rewards Eat Preferences](#21-rlvr--verifiable-rewards-eat-preferences)
22. [Constitutional AI and RLAIF](#22-constitutional-ai-and-rlaif)
23. [Reasoning RL: o1, R1, and the New Paradigm](#23-reasoning-rl-o1-r1-and-the-new-paradigm)
24. [Synthesis: RLHF → RLVR → Agent RL](#24-synthesis-rlhf--rlvr--agent-rl)

### Part III — Resources

25. [Papers, Books, Videos, Code](#25-papers-books-videos-code)
26. [Key Takeaways](#26-key-takeaways)

---

# Part I — Slide-by-Slide Companion (RL 105)

---

## 1. LLMs as Policies (Slide 2)

The lecture's opening move is the most important conceptual leap of the whole
week: **a language model is a policy**.

In Weeks 2–3 our policies were tiny networks mapping CartPole's 4-D state to
2 actions (left, right). In RLHF the policy is a 7B–700B parameter
transformer mapping a prompt $x$ to a token-level distribution over the
vocabulary $V$:

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{|y|} \pi_\theta(y_t \mid x, y_{<t})
$$

Each token $y_t$ is an action. The state at step $t$ is the prompt plus
the tokens generated so far: $s_t = (x, y_{<t})$. The episode ends at EOS or
max length.

**Two consequences this framing has that often go un-noticed**:

1. **Action space is enormous.** Vocab $|V| \approx 32$K–256K. CartPole had
   $|A| = 2$. Exploration via ε-greedy or random sampling is much harder.
2. **Reward is sparse and terminal.** You usually get one scalar reward at
   end-of-sequence ("how good was the whole response?"), not per-token
   rewards. This is why credit assignment is the central challenge of RLHF.

The MDP framing of Weeks 1–2 still applies — RLHF just inflates the action
space by 4 orders of magnitude and pushes all the reward to the last
timestep.

---

## 2. Why Fine-Tune? (Slide 3)

Pretraining maximizes $\log p_\theta(x)$ — next-token likelihood on raw
internet text. That objective rewards a model for being *plausible*, not
*helpful*. Three failure modes pretraining alone produces:

- **Continuation, not response.** Asked a question, the model continues the
  question (because internet text often does that).
- **Mode-mixing.** It picks up the average tone of all internet text, which
  is rude, hostile, and often wrong.
- **No instruction following.** Why would it? Pretraining never explicitly
  rewarded "follow the user's intent."

The HHH framing (Helpful, Harmless, Honest) from Anthropic's "A General
Language Assistant as a Laboratory for Alignment" (Askell et al., 2021) is
the post-training north star. Fine-tuning is how you get there.

---

## 3. Three Stages of LLM Training (Slide 4)

| Stage | Data | Objective | Compute share |
|-------|------|-----------|---------------|
| Pretraining | Trillions of tokens, raw text | $\max \log p_\theta(x)$ | ≈ 95–99% |
| SFT | 10K–1M instruction-response pairs | $\max \log p_\theta(y \mid x)$ | ≈ 0.5–2% |
| RLHF / RFT | Preferences or verifiable rewards | $\max \mathbb{E}[r_\phi(x,y)] - \beta \mathrm{KL}$ | ≈ 0.5–3% |

A wrinkle the slide doesn't mention: **the post-training compute share is
growing fast**. DeepSeek-R1 reportedly spent ~25% of total compute on RL
post-training. MiniMax-M1's RL stage cost \$534K (Week 4 doc). The "1%"
intuition from InstructGPT-era papers is increasingly out of date.

---

## 4. What is Reinforcement Fine-Tuning? (Slide 5)

Three components:

1. **Sample.** Generate $K$ responses from the current policy:
   $y^{(1)}, \dots, y^{(K)} \sim \pi_\theta(\cdot \mid x)$.
2. **Score.** A reward function $r(x, y)$ produces a scalar.
   - Learned reward model (RLHF)
   - Rule-based / verifier (RLVR — see §21)
   - Hybrid
3. **Update.** Apply policy gradient (PPO, GRPO, etc.) to push probability
   mass toward high-reward outputs.

The unified slogan: *RFT is fine-tuning with rewards*. RLHF is the
instance where the reward comes from a model trained on human preferences.

---

## 5. Three Levels of Abstraction (Slide 6)

The slide gives three views — let me sharpen them:

| View | What it captures | What it misses |
|------|------------------|----------------|
| Beginner | "Try answers, learn which are better" | The whole exploration / exploitation tension |
| Engineer | Sample → score → update | Why we need a KL penalty, what the value head is for |
| Researcher | Constrained optimization with KL regularization | What "preference reward model" actually is statistically |

For this doc we operate mostly at the *engineer* level, with researcher-level
asides where the math earns its keep.

---

## 6. RLHF in One Sentence (Slide 7)

> *Use human preferences as a training signal.*

The pivot is that **next-token likelihood and "humans liked this" are
different objectives**. A response can be likely (because the internet is
full of similar text) and still bad. RLHF closes that gap by training on the
"would humans prefer this?" signal directly.

The historical chain of insight:

- **2017** — Christiano et al., *Deep RL from Human Preferences* — preferences as RL reward, demonstrated on Atari and MuJoCo.
- **2020** — Stiennon et al., *Learning to Summarize from Human Feedback* — first compelling LLM RLHF result.
- **2022** — Ouyang et al., *InstructGPT* — the recipe that became ChatGPT.

---

## 7. The RLHF Pipeline (Slide 8)

The 5-step pipeline, with what each step actually costs:

1. **Collect prompts** — usually scraped or synthesized, ~10K–100K.
2. **Generate $K$ responses** ($K \in \{2, 4, 8\}$) per prompt with the SFT
   model. Forward-only, but at scale.
3. **Human ranking** — pairs $(y_w, y_l)$ where $y_w$ is preferred. The
   *expensive* step. InstructGPT used contractor labelers; modern recipes
   often use AI feedback (RLAIF, see §22).
4. **Train reward model** $r_\phi(x, y)$ on those preferences.
5. **PPO loop** — SFT model is policy + ref; reward model gives reward; KL
   penalty keeps drift bounded.

Most of the engineering complexity hides in step 5. Most of the
*data* complexity hides in step 3.

---

## 8. Reward Models (Slide 9)

The reward model is a separate network — usually initialized from the SFT
checkpoint with a scalar regression head replacing the LM head:

```
hidden = transformer(prompt + response)
last_token_hidden = hidden[:, -1, :]    # [B, hidden_dim]
reward = scalar_head(last_token_hidden) # [B, 1]
```

A few subtleties the slide doesn't get to:

- **Mean-zero centering.** During training the RM logit is centered per-batch
  to remove a shift the loss is invariant to.
- **Separate model vs. head sharing.** Training a fresh RM is the standard;
  sharing the actor's backbone is faster but invites entanglement.
- **Process Reward Models (PRMs).** Score *every step* of a chain-of-thought,
  not just the final answer. See §23.

---

## 9. Preference Modeling (Slide 10)

The lecture states the loss as:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

This is the **Bradley-Terry model** (Bradley & Terry, 1952), and it deserves
a derivation. Assume each response has an unobserved utility $r(x, y)$, and
the probability that humans prefer $y_w$ over $y_l$ is:

$$
P(y_w \succ y_l) = \frac{\exp r(x, y_w)}{\exp r(x, y_w) + \exp r(x, y_l)} = \sigma(r(x, y_w) - r(x, y_l))
$$

Maximum likelihood on observed preference data gives:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

The model only learns *relative* utility (subtract a constant from $r$ and
the loss is unchanged). That's fine — the RL stage only uses differences too.

**Where this breaks**:

- **Transitivity assumption** — Bradley-Terry assumes $A \succ B$ and
  $B \succ C$ implies $A \succ C$. Humans are not always transitive. IPO
  (§19) directly addresses this.
- **Independence of pairs** — annotator disagreement and prompt-conditional
  preferences violate it.

---

## 10. The LM as a Policy (Slide 11)

The translation table:

| Standard RL | RLHF |
|-------------|------|
| State $s$ | Prompt + tokens generated so far $(x, y_{<t})$ |
| Action $a$ | Next token $y_t$ |
| Policy $\pi(a \mid s)$ | Token distribution $\pi_\theta(y_t \mid x, y_{<t})$ |
| Trajectory | Full response $y = (y_1, \dots, y_T)$ |
| Reward | $r_\phi(x, y)$ at end of sequence (sparse!) |

One subtlety: most RLHF implementations apply the *terminal* reward at the
last token but distribute the KL penalty per-token (§12). So the per-token
"reward" for PPO is:

$$
\tilde r_t = \begin{cases} -\beta \log \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)} & t < T \\ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y_T \mid s_T)}{\pi_{\text{ref}}(y_T \mid s_T)} & t = T \end{cases}
$$

This is how PPO sees the credit assignment — terminal RM reward, per-token
KL.

---

## 11. The RLHF Objective Derived (Slide 12)

**Unconstrained version**:

$$
\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\!\left[ r_\phi(x, y) \right]
$$

This is just expected reward. Why isn't it enough? Because **maximizing a
proxy reward for too long destroys the model**. The RM is approximate; if
you push hard on it, the policy finds adversarial inputs the RM mis-scores
high (reward hacking, §12).

**KL-constrained version** — the actual objective:

$$
\max_\theta \; \mathbb{E}\!\left[ r_\phi(x, y) - \beta \, \mathrm{KL}\!\left( \pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x) \right) \right]
$$

where $\pi_{\text{ref}}$ is the SFT model (frozen). $\beta$ is the KL
coefficient — a critical hyperparameter, usually 0.01–0.2.

**Closed-form optimum** (this is the lemma DPO weaponizes — see §19):

$$
\pi^*(\cdot \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(\cdot \mid x) \exp\!\left( \frac{1}{\beta} r_\phi(x, \cdot) \right)
$$

So the optimal policy is the reference distribution *re-weighted* by an
exponential of the reward. The partition function $Z(x)$ is intractable
(sum over all sequences) — which is why we use sampling + PPO instead of
this closed form. But DPO smuggles this expression back in to derive a loss
without ever training the RM explicitly.

---

## 12. Why the KL Penalty? (Slide 13)

**Reward hacking** is the failure mode the KL penalty exists to prevent.

A concrete picture: imagine the RM gives a slightly higher score to responses
that contain the phrase "As a helpful assistant…". The unregularized policy
will start every response with that phrase, which the RM keeps rewarding,
even past the point where humans find it grating. The model has learned
*the RM*, not human preferences.

The KL penalty bounds how far the policy can drift in distribution from
$\pi_{\text{ref}}$:

$$
\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)\right]
$$

In practice it's estimated **per-token** (cheap, unbiased) using the
identity $\mathrm{KL}(p \| q) \approx \mathbb{E}_p[\log p - \log q]$ over
sampled tokens. John Schulman's "Approximating KL" trick uses
$\mathbb{E}_p[(p/q - 1) - \log(p/q)]$, which has lower variance and is
non-negative.

**Two things to watch on the dashboard**:

- **Mean KL** — should grow steadily but not explode. Sudden spikes mean reward hacking is starting.
- **KL/reward ratio** — if reward keeps climbing while KL stays flat, you're moving in the right direction. If KL outpaces reward, you're drifting.

---

## 13. PPO for RLHF (Slide 14)

The PPO clipped surrogate from Week 3:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[ \min\!\left( r_t(\theta) A_t, \; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

where $r_t(\theta) = \pi_\theta(y_t \mid s_t) / \pi_{\text{old}}(y_t \mid s_t)$.

The RLHF-specific quirks PPO has to handle:

- **Per-token actions** at vocab scale. The IS ratio can blow up violently
  when a single token has $\pi_\theta = 0.5, \pi_{\text{old}} = 0.001$.
- **Sparse terminal reward** + **dense per-token KL** means the value
  function has to bootstrap across long sequences.
- **Reference model in memory** — you need three model copies in GPU VRAM
  at training time: actor, critic, RM, ref. (Hence the popularity of LoRA
  or shared-backbone tricks.)

PPO's $\epsilon$ is usually 0.2 in RLHF, sometimes asymmetric
($\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$ — DAPO's
"clip-higher" trick from Week 4).

---

## 14. RLHF as Actor-Critic (Slide 15)

The cast of characters:

| Role | Model | Trained? | Used for |
|------|-------|----------|----------|
| Actor | $\pi_\theta$ — the LM | Yes (PPO updates) | Generating responses |
| Critic | $V_\psi(s)$ — value head | Yes (TD updates) | Estimating returns |
| Reward Model | $r_\phi$ | Frozen during RL | Scoring full responses |
| Reference | $\pi_{\text{ref}}$ — SFT model | Frozen | KL anchor |

The critic and the actor often share a backbone in production code (one
transformer, two heads). The RM and reference are *different* frozen models
held in inference-only mode.

**Memory accounting at 7B scale, fp16**:

- Actor + critic (shared backbone): ~14 GB params + ~14 GB grads + 28 GB Adam = **56 GB**
- Reference: ~14 GB (frozen, no grads)
- Reward model: ~14 GB (frozen, no grads)

Adds to ~84 GB on a single rank — which is why RLHF training usually
needs at least an 80 GB H100 or model-parallel splits.

---

## 15. The RLHF Loop in Code (Slide 16)

The slide shows pseudocode; here's a more faithful rendering:

```python
# Per RLHF iteration
prompts = sample_batch(dataset)

# 1. Rollout
with torch.no_grad():
    responses = actor.generate(prompts)              # auto-regressive sample
    old_logprobs = actor.logprobs(prompts, responses)
    ref_logprobs = reference.logprobs(prompts, responses)
    rewards = reward_model(prompts, responses)       # scalar per response
    values = critic(prompts, responses)              # per-token

# 2. Per-token effective reward (KL penalty per token)
kl = old_logprobs - ref_logprobs                     # per-token
shaped = -beta * kl
shaped[:, -1] += rewards                             # terminal RM reward

# 3. GAE
advantages, returns = gae(shaped, values, gamma, lam)

# 4. PPO epochs
for epoch in range(ppo_epochs):
    new_logprobs = actor.logprobs(prompts, responses)
    ratio = torch.exp(new_logprobs - old_logprobs)
    pg_loss = -torch.min(ratio * advantages,
                         ratio.clamp(1 - eps, 1 + eps) * advantages).mean()
    v_loss = (critic(prompts, responses) - returns).pow(2).mean()
    loss = pg_loss + 0.5 * v_loss - 0.01 * entropy
    loss.backward(); optimizer.step()
```

Compare to the CleanRL `ppo.py` we read in Week 3 — same structure, the
"environment" is now `(reward_model, reference_model)` and the actions are
tokens. Everything else is the same PPO machinery.

---

## 16. RFT vs RLHF vs SFT (Slide 17)

Expanded version of the lecture's table:

| Method | Signal | Where signal comes from | Strength | Weakness |
|--------|--------|-------------------------|----------|----------|
| SFT | Demonstrations | Human writers / model distillation | Stable, sample-efficient | Imitates exemplar distribution; can't go beyond |
| RLHF | Pairwise preferences | Human labelers on model samples | Captures nuanced "I'd rather have this" | RM brittle, reward hacking, expensive labels |
| RFT (verifiable) | Binary correctness | Code interpreter, math checker, unit tests | Cheap, scalable, no RM | Only works on verifiable domains |
| DPO | Same preference data as RLHF | Human pairs | No RM, no RL — single training pass | Can't generate new samples (fixed dataset) |
| Constitutional AI / RLAIF | AI-generated preferences | Critic LLM applies a rubric | Cheap, scaleable | Inherits critic's biases |

The 2024–2026 trend: as base models get better at being judges, the
human-label step becomes the bottleneck and AI-generated signals (RLAIF,
RLVR) increasingly dominate.

---

## 17. Why RLHF Matters (Slide 18)

The lecture lists qualities RLHF optimizes that next-token prediction
can't:

- **Helpfulness** — an answer to *the user's actual question*.
- **Instruction following** — multi-step instructions, format constraints.
- **Safety** — refuse harmful requests, decline gracefully.
- **Reasoning preference** — show work, hedge appropriately.
- **Conversational quality** — coherent, on-topic, appropriate length.

The deeper claim: *next-token likelihood under-specifies the model we want*.
There are many possible distributions all consistent with "predict the next
token in internet text" but very few that are also helpful, safe, and
honest. Post-training is how you pick out the good one.

---

## 18. Limitations (Slide 19)

What RLHF doesn't solve:

- **Noisy preferences** — labelers disagree. InstructGPT measured 73%
  agreement among trained labelers; that's the noise floor for any RM.
- **Reward model brittleness** — RMs over-fit to surface features (length,
  formatting, hedging language). The Anthropic 2023 paper *The Capacity for
  Moral Self-Correction* shows RMs can be flipped by paraphrasing.
- **Reward hacking at scale** — even with KL, hard-pushed RLHF eventually
  finds adversarial outputs.
- **Compute cost** — three (or four) model copies in VRAM. PPO is unstable.
- **Distributional alignment ≠ value alignment** — RLHF aligns to
  *averaged labeler preferences*, which is not the same thing as "what is
  good."

These limitations motivate everything in Part II — DPO removes the RM,
GRPO/CISPO stabilize PPO, RLVR removes humans, Constitutional AI removes
labelers.

---

# Part II — What's New Since the Slides

---

## 19. DPO and the RL-Free Family

**DPO** (Rafailov et al., NeurIPS 2023) is the most influential post-RLHF
paper. The trick: combine the closed-form optimum of the KL-constrained
RLHF objective (§11) with the Bradley-Terry preference model (§9), and the
reward model cancels out.

### DPO derivation (one paragraph)

Start from $\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp(r(x,y)/\beta)$.
Solve for the *implicit reward*:
$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$.
Plug into the Bradley-Terry loss; $\log Z(x)$ cancels because it appears
identically in both terms. Replace $\pi^*$ with $\pi_\theta$:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

A single supervised loss, no rollouts, no critic, no RM. Just pairs.

### The DPO family

| Variant | Year | What it changes | Why |
|---------|------|-----------------|-----|
| **DPO** | 2023 | The original | RL-free, single-pass |
| **IPO** | 2024 | Replaces $\log\sigma(\cdot)$ with squared loss | Fixes DPO over-fitting on near-deterministic preferences |
| **KTO** | 2024 | Uses prospect-theory loss; works on *unpaired* binary labels | No need for $(y_w, y_l)$ pairs |
| **ORPO** | 2024 | Combines SFT and preference loss in *one stage*; no reference model | Cuts compute, decouples from $\pi_{\text{ref}}$ |
| **SimPO** | 2024 | Uses average log-prob as implicit reward; reference-free | Lighter, often better than DPO |
| **cDPO** | 2024 | Conservative DPO — handles label noise | Real preference data is noisy |
| **R-DPO** | 2024 | Length-regularized DPO | DPO has a length-bias failure mode |
| **DPOP** | 2024 | DPO with positive samples only | When you only have "good" examples |

Practical 2025 stack (per the survey literature): **SimPO** for the broad
alignment pass, **ORPO** when you can train SFT + preference jointly,
**KTO** when you have only thumbs-up/down (no pairs), **DPO** as the
fallback baseline.

### Key trade-off DPO has vs. PPO-RLHF

DPO is **offline**: it trains on a fixed preference dataset. It cannot
explore. If the dataset misses a region of the response space, DPO can't
find it. PPO-RLHF *generates new samples each iteration* and asks the RM,
so it explores. This is why high-quality RLHF often outperforms DPO at
scale — but it costs an order of magnitude more compute.

---

## 20. Post-PPO: GRPO, DAPO, CISPO

Week 4 covers this lineage in depth — here's the compressed map.

### GRPO (DeepSeekMath, Feb 2024)

Drops the critic entirely. Uses **group-relative advantage**: sample $K$
responses per prompt, compute reward, normalize:

$$
A_i = \frac{r_i - \mathrm{mean}(r_{1..K})}{\mathrm{std}(r_{1..K})}
$$

No value function, no GAE. Cuts memory by ~33% (no critic = one fewer
model copy). **DeepSeek-R1's primary algorithm.**

### DAPO (ByteDance, Mar 2025)

Four practical fixes over GRPO:

1. **Clip-Higher** — asymmetric clip ($\epsilon_{\text{high}} = 0.28 > \epsilon_{\text{low}} = 0.2$) to give entropy room to grow.
2. **Dynamic Sampling** — re-sample prompts where every response gets the same reward (zero-advantage prompts contribute nothing).
3. **Token-Level PG Loss** — sum, don't mean, over response length.
4. **Overlong Reward Shaping** — soft penalty for responses that hit max length.

### CISPO (MiniMax-M1, Jun 2025)

Stop-gradient on the importance-sampling ratio:

$$
\mathcal{J}_{\text{CISPO}}(\theta) = \mathbb{E}\!\left[ \frac{1}{\sum |o_i|} \sum_{i,t} \mathrm{sg}(\hat r_{i,t}) \, \hat A_{i,t} \, \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]
$$

The IS ratio still *clips* the gradient magnitude, but the gradient flows
through the log-prob term only — not through the ratio itself. This
absorbs MoE-router-induced variance and made the M1 training stable enough
to run for 16 off-policy updates per batch.

**Lineage summary** (with full derivations in the Week 4 doc):

| Algorithm | Year | Key change | Where it shines |
|-----------|------|------------|-----------------|
| PPO | 2017 | Clipped surrogate | General RL |
| PPO-RLHF | 2022 | + KL penalty + RM | InstructGPT/ChatGPT |
| GRPO | 2024 | Drop critic, group-relative advantage | DeepSeek-R1 (math/code) |
| DAPO | 2025 | Clip-higher, dynamic sampling, token-level loss | Reasoning RL at scale |
| CISPO | 2025 | Stop-grad IS ratio | MoE-RL, agent RL |

---

## 21. RLVR — Verifiable Rewards Eat Preferences

**RLVR** (Reinforcement Learning with Verifiable Rewards) replaces the
learned RM with a *verifier function*: a piece of code that returns 0/1.

Where this works:

- **Math** — does the final answer match the ground truth?
- **Code** — do the unit tests pass?
- **Logic puzzles** — does the solver agree?
- **Format** — does the response match a regex?
- **Tool use** — did the API call succeed?

Where it doesn't:

- Subjective quality (helpfulness, tone, safety)
- Long-form writing
- Anything where "correct" is a judgment call

### The DeepSeek-R1 moment

DeepSeek-R1 (Jan 2025) demonstrated that **GRPO + verifiable rewards on
base model** produces emergent chain-of-thought reasoning *without SFT
warmup*. R1-Zero was trained from the base model with no instruction
tuning, only verifiable math and code rewards. It learned to "think out
loud" on its own. This was the first widely-replicable open result that
strong reasoning emerges from RLVR alone.

### Tülu 3 (Allen AI, Nov 2024)

The first open-recipe reproduction. Pipeline: SFT → DPO → RLVR with PPO
on math/code/IFEval verifiable signals. Shipped open weights, open data,
open hyperparameters. Set a new bar for "reproducible alignment."

### "Faster, not smarter"

A critical 2025 finding (from work like Yue et al., Promptfoo's analysis,
and others): RLVR on base models often *amplifies* reasoning patterns the
base already had, rather than teaching new ones. The base model could
already get the right answer with high enough sampling temperature; RLVR
re-weights toward those trajectories. The implication: RLVR is more like
"sharpening the policy on its own best trajectories" than "teaching new
skills." This re-frames the gains as efficiency, not capability.

This is good news (it's cheaper than expected) and bad news (the ceiling
is the base model's capability).

### Why verifiable beats preferences for reasoning

- **Zero label noise.** A solution is right or wrong.
- **Zero reward hacking.** No RM to fool — the verifier is exact.
- **Cheap.** No labelers, no RM training, no RM serving.
- **Process-aligned.** A wrong answer is a wrong answer; no surface-feature
  bias.

Limit: only 5–10% of LLM use cases are verifiable. Outside that band,
preferences (and DPO/PPO-RLHF) still rule.

---

## 22. Constitutional AI and RLAIF

**Constitutional AI** (Anthropic, 2022): replace human preference labelers
with an AI critic guided by a written "constitution." The loop:

1. Sample model response.
2. Critic LLM rates the response against constitutional principles.
3. Critic also generates a *revision* that scores higher.
4. Train on the revised response (SFT) or the (original, revised) pair as a preference (RLAIF).

**Why this works**: a sufficiently capable model can apply a written rubric
more consistently than a human can. *Reproducibility beats accuracy* for
training signal.

**RLAIF** (Bai et al., 2022; Lee et al., 2023): RLHF with AI-generated
preferences. Same machinery as RLHF, just AI labels. Works almost as well
on standard benchmarks at a fraction of the cost.

**The 2025 picture**: most production post-training pipelines use a *mix*
of human preferences (for the most subjective decisions), AI preferences
(for scale), and verifiable rewards (for math/code/format). Pure-human
RLHF is increasingly rare outside frontier safety work.

---

## 23. Reasoning RL: o1, R1, and the New Paradigm

Late 2024 brought a phase transition: **reasoning RL**. Models trained to
produce long chain-of-thought before answering, with RL on the *final
answer* (verifiable) rather than the *reasoning process*.

### OpenAI o1 (Sep 2024)

First public model in the new paradigm. Long internal reasoning ("thinking
tokens") visible in the API as a hidden trace, with summary returned to
the user. OpenAI hasn't published the recipe but the public information is
consistent with: large-scale RL on verifiable rewards + reasoning data
distillation.

### DeepSeek-R1 (Jan 2025)

Open replication. Two versions:

- **R1-Zero**: pure RL from base, no SFT. Discovered "Aha moments" emerge from RL alone.
- **R1**: cold-start SFT → RL → SFT → RL. Cleaner outputs.

GRPO + rule-based rewards (math correctness, code unit tests, format
compliance). 671B MoE base, only ~37B active.

### Process Reward Models (PRMs) vs Outcome Reward Models (ORMs)

- **ORM** — score the final answer only. RLVR is essentially ORM-driven.
- **PRM** — score *every step* of the reasoning. Lightman et al. 2023
  ("Let's Verify Step by Step") showed PRMs outperform ORMs on math.
  But PRMs require step-level labels, which are expensive.

The current consensus: ORMs scale better when correctness is verifiable;
PRMs are useful when you need to credit-assign a wrong reasoning chain.

---

## 24. Synthesis: RLHF → RLVR → Agent RL

The arc the slides hint at and Part II makes explicit:

```
2017  Christiano: preferences as RL signal
2020  Stiennon: LLM RLHF on summarization
2022  InstructGPT: RLHF recipe → ChatGPT
2023  DPO: RL-free preference learning
2024  GRPO + RLVR: reasoning emerges from verifiable rewards
2025  DAPO/CISPO: stable RL on long contexts and MoE
2025+ Agent RL: RL on tool-use, multi-turn, real environments (Week 4)
```

Three under-the-hood shifts make this story coherent:

1. **The reward source moved**: human → RM → DPO (no RM) → verifier → environment.
2. **The compute moved**: pretraining-dominant → SFT-heavy → RL-heavy.
3. **The policy moved**: instruction-following LM → reasoning model → agent.

We started Week 1 with CartPole and Bellman. We end Week 6 with frontier
post-training. The math hasn't changed — the policies got bigger and the
rewards got harder to specify.

---

# Part III — Resources

---

## 25. Papers, Books, Videos, Code

### Foundational papers (read these in order)

1. **Christiano et al. 2017** — *Deep Reinforcement Learning from Human Preferences* — [arxiv:1706.03741](https://arxiv.org/abs/1706.03741). The origin.
2. **Ziegler et al. 2019** — *Fine-Tuning Language Models from Human Preferences* — [arxiv:1909.08593](https://arxiv.org/abs/1909.08593). First LM RLHF.
3. **Stiennon et al. 2020** — *Learning to Summarize from Human Feedback* — [arxiv:2009.01325](https://arxiv.org/abs/2009.01325). The first compelling result.
4. **Ouyang et al. 2022** — *InstructGPT* — [arxiv:2203.02155](https://arxiv.org/abs/2203.02155). The recipe behind ChatGPT.
5. **Bai et al. 2022** — *Constitutional AI* — [arxiv:2212.08073](https://arxiv.org/abs/2212.08073). RLAIF foundations.

### The post-RLHF wave

6. **Rafailov et al. 2023** — *DPO* — [arxiv:2305.18290](https://arxiv.org/abs/2305.18290).
7. **Ethayarajh et al. 2024** — *KTO* — [arxiv:2402.01306](https://arxiv.org/abs/2402.01306).
8. **Azar et al. 2023** — *IPO* — [arxiv:2310.12036](https://arxiv.org/abs/2310.12036).
9. **Hong et al. 2024** — *ORPO* — [arxiv:2403.07691](https://arxiv.org/abs/2403.07691).
10. **Meng et al. 2024** — *SimPO* — [arxiv:2405.14734](https://arxiv.org/abs/2405.14734).
11. **Xu et al. 2024** — *RainbowPO survey* — [arxiv:2410.04203](https://arxiv.org/abs/2410.04203).

### Reasoning RL & RLVR

12. **DeepSeekMath / GRPO** — [arxiv:2402.03300](https://arxiv.org/abs/2402.03300).
13. **DeepSeek-R1** — [arxiv:2501.12948](https://arxiv.org/abs/2501.12948).
14. **DAPO** — [arxiv:2503.14476](https://arxiv.org/abs/2503.14476).
15. **CISPO / MiniMax-M1** — [arxiv:2506.13585](https://arxiv.org/abs/2506.13585).
16. **Tülu 3** (Allen AI) — [arxiv:2411.15124](https://arxiv.org/abs/2411.15124).
17. **Lightman et al. 2023** — *Let's Verify Step by Step* (PRMs) — [arxiv:2305.20050](https://arxiv.org/abs/2305.20050).

### Books

- **Nathan Lambert** — *Reinforcement Learning from Human Feedback* (Interconnects Press) — the only end-to-end RLHF textbook. [rlhfbook.com](https://rlhfbook.com/)
- **Sutton & Barto** — *Reinforcement Learning: An Introduction*, 2nd ed. — chapters 13 and 17 ground the policy gradient and trust-region material RLHF builds on.

### Blog posts (start here if papers are too dense)

- **HuggingFace** — *Illustrating RLHF* — [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf). The canonical visual explainer.
- **Sebastian Raschka** — *LLM Training: RLHF and Its Alternatives* — [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives).
- **Sebastian Raschka** — *State of LLMs 2025* — [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/p/state-of-llms-2025). Yearly state-of-the-field summary.
- **Nathan Lambert** — *Interconnects* — [interconnects.ai](https://www.interconnects.ai/). The best newsletter for tracking the post-RLHF research front.
- **John Schulman** — *Approximating KL Divergence* — [joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html). Why we use the $k_3$ estimator in PPO-RLHF.

### Videos

- **Andrej Karpathy — State of GPT (2023)** — [youtube.com/watch?v=bZQun8Y4L2A](https://www.youtube.com/watch?v=bZQun8Y4L2A). The 40-min overview of pretraining → SFT → RM → PPO that everyone cites.
- **Andrej Karpathy — Deep Dive into LLMs (2024)** — [youtube.com/watch?v=7xTGNNLPyMI](https://www.youtube.com/watch?v=7xTGNNLPyMI). The successor; covers RLHF, RLVR, and o1-style reasoning.
- **Hugging Face Deep RL Course — Bonus Unit on RLHF** — [huggingface.co/learn/deep-rl-course/unitbonus3/rlhf](https://huggingface.co/learn/deep-rl-course/unitbonus3/rlhf).
- **AI Scholars — RL 101 Past Sessions** — [YouTube playlist](https://www.youtube.com/watch?v=4e0laDA7jlM&list=PLte0_KfXCwoh2EX7KRmooLU-Jyn-y8BQZ). Our own study group's prior weeks.

### Code (read these to internalize)

- **HuggingFace TRL** — [github.com/huggingface/trl](https://github.com/huggingface/trl). Reference implementation of PPO-RLHF, DPO, GRPO, KTO, ORPO. The "CleanRL of RLHF."
- **OpenRLHF** — [github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). High-throughput RLHF framework; supports Ray + vLLM rollouts.
- **verl** (ByteDance) — [github.com/volcengine/verl](https://github.com/volcengine/verl). The framework DAPO was developed in.
- **NVIDIA NeMo-Aligner** — [github.com/NVIDIA/NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner). Production-scale RLHF.

### Companion docs in this repo

- **Week 4** — [`docs/week4-agent-rl-forge.md`](week4-agent-rl-forge.md) — agent RL systems, full PPO→GRPO→DAPO→CISPO derivations.
- **Week 5** — [`docs/week5-minimax-m27-visual-guide.md`](week5-minimax-m27-visual-guide.md) — full RLHF stack with M2.7 case study, RLVR, Constitutional AI, DPO closed-form derivation.

---

## 26. Key Takeaways

1. **An LLM is a policy.** Tokens are actions. Sequences are trajectories. Everything from Week 2 onward applies; the only thing that scales is the action space and the cost of one rollout.
2. **The RLHF objective is KL-constrained reward maximization.** The KL penalty is what keeps reward hacking bounded. The closed-form optimum of this objective is what DPO weaponizes.
3. **The Bradley-Terry preference loss only learns relative utility.** That's enough — RL only ever uses reward differences.
4. **DPO turns RLHF into a single supervised pass** by collapsing the closed-form optimum + Bradley-Terry into one loss. Cheaper, less explorative.
5. **GRPO drops the critic.** DAPO adds practical fixes. CISPO stop-gradients the IS ratio. Each generation is one fix on top of the last; nothing is from scratch.
6. **RLVR replaces the RM with a verifier.** Cheaper, no reward hacking, but only works on verifiable domains.
7. **Constitutional AI / RLAIF replace human labelers with AI critics.** Most production stacks now blend human, AI, and verifiable signals.
8. **Reasoning RL (o1, R1) is the current frontier.** Long chain-of-thought + RLVR on the final answer. Surprisingly, reasoning emerges from RL on a base model with no SFT (R1-Zero).
9. **The trajectory of post-training: human → RM → DPO → verifier → environment.** Each step removes a labeling bottleneck and lets compute scale further.
10. **Modern LLM alignment is reinforcement learning in disguise.** And increasingly, *not* in disguise.

---

*RL 101 Study Group — Colby Ziyu Wang @ SparkCraft / Hosted by AI Scholars*
*Notes: Alp Guneysel*

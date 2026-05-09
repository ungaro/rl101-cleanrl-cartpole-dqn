---
marp: true
theme: default
paginate: true
header: "RL 101 — Week 6 — RLHF & Reinforcement Fine-Tuning Under the Hood"
footer: "rl101-crash-course"
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

### Part III — Common Questions Across All Weeks

25. [Q&A — Foundations and Value Methods (Weeks 1–2)](#25-qa--foundations-and-value-methods-weeks-12)
26. [Q&A — PPO and Actor-Critic (Week 3)](#26-qa--ppo-and-actor-critic-week-3)
27. [Q&A — Agent RL and Systems (Week 4)](#27-qa--agent-rl-and-systems-week-4)
28. [Q&A — RLHF, DPO, RLVR (Weeks 5–6)](#28-qa--rlhf-dpo-rlvr-weeks-56)
29. [Q&A — Practical and Big Picture](#29-qa--practical-and-big-picture)

### Part IV — Community Pulse

30. [What the Community Is Debating Right Now](#30-what-the-community-is-debating-right-now)

### Part V — Where to Go from Here

31. [Specialization Tracks](#31-specialization-tracks)
32. [Hands-On Milestones](#32-hands-on-milestones)
33. [Open Research Questions](#33-open-research-questions)
34. [Communities and Staying Sharp](#34-communities-and-staying-sharp)

### Part VI — Resources

35. [Papers, Books, Videos, Code](#35-papers-books-videos-code)
36. [Key Takeaways](#36-key-takeaways)

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

```math
\pi_\theta(y \mid x) = \prod_{t=1}^{|y|} \pi_\theta(y_t \mid x, y_{\lt t})
```

Each token $y_t$ is an action. The state at step $t$ is the prompt plus
the tokens generated so far: $s_t = (x, y_{\lt t})$. The episode ends at EOS or
max length.

**Two consequences this framing has that often go un-noticed**:

1. **Action space is enormous.** Vocab $|V| \approx 32\text{K}\text{–}256\text{K}$. CartPole had
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
| State $s$ | Prompt + tokens generated so far $(x, y_{\lt t})$ |
| Action $a$ | Next token $y_t$ |
| Policy $\pi(a \mid s)$ | Token distribution $\pi_\theta(y_t \mid x, y_{\lt t})$ |
| Trajectory | Full response $y = (y_1, \dots, y_T)$ |
| Reward | $r_\phi(x, y)$ at end of sequence (sparse!) |

One subtlety: most RLHF implementations apply the *terminal* reward at the
last token but distribute the KL penalty per-token (§12). So the per-token
"reward" for PPO is:

```math
\tilde r_t = \begin{cases} -\beta \log \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)} & t \lt T \\ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y_T \mid s_T)}{\pi_{\text{ref}}(y_T \mid s_T)} & t = T \end{cases}
```

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

```math
\pi^{\ast}(\cdot \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(\cdot \mid x) \exp\!\left( \frac{1}{\beta} r_\phi(x, \cdot) \right)
```

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

Start with the closed-form optimum from §11:

```math
\pi^{\ast}(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp(r(x,y)/\beta)
```

Solve for the *implicit reward*:

```math
r(x, y) = \beta \log \frac{\pi^{\ast}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
```

Plug into the Bradley-Terry loss; $\log Z(x)$ cancels because it appears
identically in both terms. Replace $\pi^{\ast}$ with $\pi_\theta$:

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

# Part III — Common Questions Across All Weeks

These are the questions that come up over and over when people first work
through this material. Answers are deliberately short — for full
treatments, follow the §-references back into the doc.

---

## 25. Q&A — Foundations and Value Methods (Weeks 1–2)

**Q1. Why does RL need a discount factor $\gamma$ if my episodes are short?**
$\gamma < 1$ does two things even on short episodes: (i) makes the infinite-horizon sum mathematically finite, and (ii) gives the agent a "preference for now" — a bird in the hand vs. two in the bush. On CartPole specifically you can almost get away with $\gamma=1$ because episodes terminate in ≤500 steps; in practice $\gamma=0.99$ is just a smoother gradient signal.

**Q2. On-policy vs. off-policy — what's the actual difference and why does it matter?**
On-policy = you can only learn from data sampled by the *current* policy (PPO). Off-policy = you can learn from data sampled by *any* policy (DQN, GRPO with replay). Off-policy is sample-efficient (re-use old data) but unstable (importance sampling ratios blow up). On-policy is stable but wasteful — every gradient step throws away the rollout. The post-PPO algorithms in §20 are all attempts to soften this trade.

**Q3. Is the Bellman equation a definition or a theorem?**
A definition. $Q(s,a) = \mathbb{E}[r + \gamma \max_{a'} Q(s',a')]$ is just unrolling the value function one step. The *useful* claim is that the iteration $Q_{k+1} = T Q_k$ (where $T$ is the Bellman operator) is a contraction in $\ell_\infty$ norm — that's the theorem, and it's why value iteration converges.

**Q4. Why does DQN need a target network?**
Because without one, you're regressing $Q_\theta(s,a)$ toward a target that's *also computed from* $Q_\theta$. The target moves every gradient step. The target network is a stale copy of $Q_\theta$ updated every $N$ steps (usually 500–10K) so the target stays still long enough for the regression to converge before it shifts.

**Q5. Can I use DQN on LLMs?**
In principle yes — actions are tokens, $|A| = |V|$. In practice no: the action space is 50–250K, $\max_a Q(s,a)$ is a 50K-way max at every step, and Q-functions over discrete tokens are hard to learn at that scale. Policy gradient methods (PPO, GRPO) sample from $\pi_\theta$ directly and bypass the max.

---

## 26. Q&A — PPO and Actor-Critic (Week 3)

**Q6. Why does PPO clip instead of using a trust region like TRPO?**
TRPO solves a constrained optimization at every step ($\mathrm{KL} \le \delta$) using conjugate gradient and a line search. PPO approximates the same idea with a clip — much simpler to implement, almost as good in practice. PPO's clip is a *first-order* surrogate for TRPO's exact trust region.

**Q7. Why doesn't PPO use a replay buffer like DQN?**
PPO is on-policy. The clipped surrogate's correctness depends on the data being sampled from $\pi_{\text{old}}$, which is at most a few gradient steps stale. A replay buffer breaks that assumption. PPO does re-use rollouts within an "update epoch" (usually 4–10 epochs per rollout) — that's the closest it gets.

**Q8. What does GAE actually buy over Monte Carlo returns?**
A bias-variance knob. Pure Monte Carlo ($\lambda=1$) is unbiased but high variance. Pure 1-step TD ($\lambda=0$) is low variance but biased. GAE's $\lambda \in [0,1]$ blends them. $\lambda=0.95$ is the standard PPO default — empirically a sweet spot.

**Q9. Is $\epsilon=0.2$ in PPO theory or empirical?**
Empirical. The original PPO paper swept it; 0.2 was best on Atari/Mujoco. Some modern variants like DAPO use asymmetric clips ($\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}=0.28$) to give entropy room — see §20.

---

## 27. Q&A — Agent RL and Systems (Week 4)

**Q10. Why does agent RL need 100K+ environments — can't we just have one good one?**
One environment over-fits. The whole point of the diversity is that the policy generalizes across tasks, tools, and contexts. CartPole-RL teaches you "balance the pole." Agent-RL teaches you "use any tool." That requires variety the same way pretraining requires variety.

**Q11. What does "200K-token context" mean for the actor — is it loaded once?**
Once per rollout. The actor sees the full prompt + tool-call history + responses as it generates. The KV cache holds the entire context during decode; this is why RL on long-context LLMs is memory-bound, not compute-bound. Hence prefix-tree training (Magi Attention) and PD disaggregation in Week 4.

**Q12. Why does CISPO stop-gradient the IS ratio — isn't that information loss?**
Yes — and that's the point. The IS ratio explodes when $\pi_\theta$ and $\pi_{\text{old}}$ disagree heavily on a single token (which happens constantly in MoE-RL because the router routes the *same* token to different experts at different steps). Stop-gradient keeps the *magnitude* of the gradient sane while preserving the gradient *direction* through $\log \pi_\theta$. It's a variance-reduction trick at the cost of a small bias.

**Q13. Can I do agent RL on a 7B model, or is this strictly a frontier-lab thing?**
Yes, you can. Tülu 3 ships an 8B variant. Several open frameworks (TRL's GRPO trainer, OpenRLHF, verl) work on a single 80GB H100 with LoRA. The frontier-lab thing is *long-context* agent RL with hundreds of tools — that needs the systems work in Week 4. Plain GRPO on math/code with a 7B model is hobbyist-tier in 2025–2026.

---

## 28. Q&A — RLHF, DPO, RLVR (Weeks 5–6)

**Q14. Why a reward model — can't humans just label every response during training?**
Two reasons. Speed: PPO updates 100K+ times; humans can't label that fast. Cost: a labeler costs ~\$50/hr; an RM forward pass costs ~\$0.0001. The RM is a *learned approximation* of human preferences that scales.

**Q15. Why does the RM output a scalar instead of a distribution?**
Because the Bradley-Terry preference model only requires a scalar utility per response — see §9. A distribution would over-specify the problem; we don't ask "how good with what uncertainty?" we ask "did humans prefer A or B?" The pairwise margin is enough.

**Q16. Why does the KL penalty target the SFT model specifically and not pretraining?**
Two reasons. (1) Pretraining has the wrong format — it doesn't follow instructions. The SFT model is the closest "fluent + on-task" anchor. (2) KL to pretraining is huge; you'd need an enormous $\beta$ to bound it, killing learning. SFT is the natural reference for "stay in this neighborhood while improving."

**Q17. If DPO is so much simpler than PPO-RLHF, why is anyone still using PPO?**
Three reasons: (i) DPO is offline — it can't generate new samples, so it's bottlenecked by the dataset's coverage; (ii) PPO can use richer reward signals (rule-based + RM + verifiable mixed); (iii) at frontier scale, PPO-RLHF still slightly outperforms DPO on hard preference tasks where the response distribution moves a lot during training.

**Q18. How do I pick between DPO / IPO / KTO / ORPO / SimPO?**
Quick decision tree (see §19 for detail):
- *Just paired preferences, want simplest baseline*: DPO
- *Paired preferences, want best results*: SimPO (reference-free)
- *Paired preferences, near-deterministic*: IPO (avoids the DPO over-fit)
- *Only thumbs up/down (no pairs)*: KTO
- *Want SFT + preference in one stage*: ORPO
- *Production stack*: SimPO for the broad pass, KTO for off-policy "bad example" pinning

**Q19. Why does RLVR work for math but not creative writing?**
Verifiability. A math answer is right or wrong; a code unit test passes or fails. "Was this story good?" has no verifier. RLVR is bounded to the 5–10% of tasks that have a binary correctness criterion. Outside that band, you still need preference data.

---

## 29. Q&A — Practical and Big Picture

**Q20. How do I debug an RL run that's diverging?**
Standard checklist:
1. Plot per-step KL divergence — sudden spikes mean reward hacking.
2. Plot reward distribution — if it saturates or collapses to a few modes, your RM/policy is degenerate.
3. Plot entropy — if it crashes, the policy is over-committing.
4. Lower learning rate by 3×, then 10×.
5. Check chat template / tokenizer is identical between policy and reference. (This breaks more runs than any algorithm bug — see §30.)
6. Verify reward direction (sign flips happen).
7. Re-check that gradient norms aren't blowing up.

**Q21. How do I pick the KL coefficient $\beta$?**
$\beta$ controls a trade-off: higher $\beta$ → stays close to SFT, slower learning; lower $\beta$ → faster learning, more reward hacking. Standard range: 0.01–0.2. Start at 0.05, watch the KL/reward ratio. If reward grows faster than KL, you can lower $\beta$. If KL outpaces reward, raise $\beta$. Some recipes (DAPO) drop the explicit penalty entirely and rely on the clip alone.

**Q22. What's the smallest model I can do RLHF on?**
You can run DPO on a 350M-1B model on a single 24GB GPU with LoRA. Full PPO-RLHF (4 model copies) starts being practical at 7B on an 80GB H100. RLVR / GRPO is the cheapest variant — DeepSeek-R1's 1.5B and 7B distilled checkpoints were trained on small clusters.

**Q23. How do I evaluate an RLHF model — automated benchmarks aren't enough, are they?**
Right, they're not. Use a stack: (i) automated benchmarks (MMLU, GSM8K, IFEval) for sanity; (ii) Arena-Hard / AlpacaEval 2 / MTBench for LLM-judged head-to-heads; (iii) **a held-out human eval set** for what you actually care about; (iv) red-team prompts for safety. Skipping (iii) is the most common mistake — automated metrics over-fit fast.

**Q24. Is RL on LLMs just curve fitting in disguise?**
Half-true. RLVR on a base model mostly *re-weights* trajectories the model could already produce (the "faster, not smarter" finding, §21) — that *is* curve fitting. But when the policy actually generates novel reasoning chains and gets rewarded for them (R1-Zero's "Aha moments"), you're past pure interpolation. The honest answer: closer to curve fitting than people pretend, but not entirely.

**Q25. Will reasoning RL scale like pretraining did, or hit a wall?**
Open question. Pretraining had clean scaling laws because loss is smooth. RLVR has bumpier dynamics — reward hacking, mode collapse, capability ceilings tied to the base. Early scaling results are encouraging but extrapolation is much shakier than for pretraining. Watch papers like *RLVR scaling laws* (2025) for the actual curves.

**Q26. What's the strongest argument that RLHF doesn't actually align models?**
RLHF aligns to *averaged labeler preferences*, which is not the same thing as "what is good." If labelers consistently prefer hedging language, your RM rewards hedging, and your policy hedges — even when the user wants a direct answer. Plus reward hacking is provably present at scale (see Anthropic's *Capacity for Moral Self-Correction* and OpenAI's reward-hacking studies). RLHF is a useful *approximation* to alignment, not alignment.

---

# Part IV — Community Pulse

---

## 30. What the Community Is Debating Right Now

A snapshot of what r/reinforcementlearning, r/LocalLLaMA, and
r/MachineLearning are actively discussing in the months around this
lecture (Apr–May 2026). The doc above explains the algorithms; this
section is the *community state* you're stepping into when you go ship
something.

### Cross-cutting themes that map directly to this course

1. **"PPO vs. GRPO vs. DPO ranking depends entirely on hyperparameter tuning."** The canonical thread is [I implemented PPO, GRPO, and DPO from scratch and the ranking completely reversed after hyperparameter tuning](https://reddit.com/r/reinforcementlearning/comments/1sc9d0y/). This is the single most important practical lesson and reinforces §28 Q18: there is no universal winner; algorithms differ in *what they tolerate*, not in absolute quality.

2. **Chat-template and tokenizer hygiene silently destroys fine-tunes.** Recurring posts: [PSA: Qwen3.6 ships with `preserve_thinking` — make sure you have it on](https://reddit.com/r/LocalLLaMA/comments/1sne4gh/) (413 pts), [it's time to update your Gemma 4 GGUFs (chat template fixed)](https://reddit.com/r/LocalLLaMA/comments/1t3dfvp/). This is the single most common reason DPO/RLHF runs collapse in practice — the policy and reference disagree on prompt formatting, KL explodes, training diverges. Always verify template equality before you debug the algorithm. (See §29 Q20 step 5.)

3. **Reward shaping / reward hacking is the universal RL pain point.** Spans pure-RL ([PPO rewards crashing mid-training on Pendulum](https://reddit.com/r/reinforcementlearning/comments/1t4lbl3/), [Reward STD collapse](https://reddit.com/r/reinforcementlearning/comments/1t2ykr5/)) to RLHF (reward-model over-optimization, [Heretic / abliteration ecosystem](https://reddit.com/r/LocalLLaMA/comments/1sw77p0/) as the post-RLHF safety reversal). RLVR is the community's escape hatch where it applies; outside that band you live with reward shaping.

4. **Agent / RL framework sprawl.** The active debate of "which stack do I use" resolves to: *TRL (HuggingFace)* for prototyping, *verl (ByteDance)* for production-scale RL, *OpenRLHF* for high-throughput Ray + vLLM rollouts, *SGLang* or *vLLM* for inference. See [What standard RL frameworks do people use these days?](https://reddit.com/r/reinforcementlearning/comments/1szkr2m/) for the working consensus. We list these in §35.

5. **"Is X still worth it in 2026?" identity crises.** [Is DQN still worth in 2026?](https://reddit.com/r/reinforcementlearning/comments/1srpz6h/), questions about value of ML PhDs, whether local LLMs are worth running. Frame: nothing in this course is dead — DQN is still the cleanest way to *learn* value-based RL, even if frontier deployment uses GRPO.

### Per-subreddit highlights

**r/reinforcementlearning** — heavily focused on practical pain (reward shaping, PPO instability, framework choice) and the RL-meets-LLM-agents transition ([How RL fits into tool-using LLM agents](https://reddit.com/r/reinforcementlearning/comments/1sjpho5/), [Project: I gave an LLM memory of its own mistakes](https://reddit.com/r/reinforcementlearning/comments/1t46tyy/)). Single-author serialized projects (like a [3-Mac-Mini GRPO experiment](https://reddit.com/r/reinforcementlearning/comments/1t49nlz/)) are popular.

**r/LocalLLaMA** — model-release-driven (Qwen 3.6, Gemma 4, MiniMax M2.7), with a strong sub-current of *uncensoring / abliteration* as the inverse of RLHF. The [MiniMax M2.7 release thread](https://reddit.com/r/LocalLLaMA/comments/1sj0dm3/) (675 pts) and [the license discussion](https://reddit.com/r/LocalLLaMA/comments/1skabyf/) are direct Week 4–5 reading. The community here mostly *uses* fine-tuned models rather than training them.

**r/MachineLearning** — meta-debates dominate: [conference lottery culture](https://reddit.com/r/MachineLearning/comments/1t0mct7/), [reproducibility crises](https://reddit.com/r/MachineLearning/comments/1sml5fo/). On topic, [Studying Sutton & Barto and its connections to RL for LLMs](https://reddit.com/r/MachineLearning/comments/1sgknct/) is *exactly* the bridge this course builds. [DeepSeek V4 FP4 QAT details](https://reddit.com/r/MachineLearning/comments/1t7yrvr/) is the other thread to read.

### High-signal reading list (8 threads)

If you read nothing else from these subreddits, read these:

1. [I implemented PPO, GRPO, and DPO from scratch — ranking completely reversed after hyperparameter tuning](https://reddit.com/r/reinforcementlearning/comments/1sc9d0y/) — the canonical "algorithms < hyperparameters" thread. **Weeks 3–6.**
2. [Why is PPO still the de facto RL algorithm for LLM training?](https://reddit.com/r/reinforcementlearning/comments/1mo9guy/) — 26 nuanced takes from practitioners. **Weeks 5–6.**
3. [Is DQN still worth in 2026?](https://reddit.com/r/reinforcementlearning/comments/1srpz6h/) — direct Week 2 community pulse.
4. [What standard RL frameworks do people use these days? (TRL/verl/openRLHF/sglang)](https://reddit.com/r/reinforcementlearning/comments/1szkr2m/) — Week 4 stack-picking guide.
5. [How RL fits into tool-using LLM agents](https://reddit.com/r/reinforcementlearning/comments/1sjpho5/) — clean Week 4 framing question.
6. [Studying Sutton & Barto and its connections to RL for LLMs](https://reddit.com/r/MachineLearning/comments/1sgknct/) — the exact bridge this course is.
7. [PSA: Qwen3.6 ships with `preserve_thinking`](https://reddit.com/r/LocalLLaMA/comments/1sne4gh/) — the cautionary tale on chat-template hygiene.
8. [Heretic plagiarism / abliteration ecosystem](https://reddit.com/r/LocalLLaMA/comments/1sw77p0/) — entry point to the post-RLHF "uncensoring" debate, which is its own ethics rabbit hole.

---

# Part V — Where to Go from Here

This is the last lecture of the series. Six weeks ago we were balancing a
pole. Today we're at the frontier of how every modern AI system gets
trained. Here's how to keep going.

---

## 31. Specialization Tracks

Pick one of these based on what energized you most. *Don't try to do all
three at once* — depth in one is more valuable than shallow coverage of
all.

### Track A — Classical RL (if Weeks 2–3 hit hardest)

The deepening path:
- **Sutton & Barto chapters 6–13** — TD methods, function approximation, policy gradient theorem.
- **CleanRL's full algorithm zoo** — DDPG, TD3, SAC for continuous control; PPO-LSTM for partial observability; the Atari PPO variant for high-dim observations.
- **OpenAI Spinning Up** — [spinningup.openai.com](https://spinningup.openai.com/) — the canonical hands-on tutorial.
- **Hands-on**: solve LunarLanderContinuous-v3, then BipedalWalker-v3, then a custom task in Mujoco / Isaac Gym.
- **Frontier**: world-model RL (DreamerV3, MuZero, V-JEPA-2), offline RL (CQL, IQL, AWAC), exploration (RND, ICM, NoisyNets).

### Track B — Agent / LLM RL (if Week 4 hit hardest)

The deepening path:
- **Read the primary papers in full**: MiniMax-M1 (Forge/CISPO), DeepSeek-R1, Tülu 3, DAPO. These are dense; budget 2 hours each.
- **Code tour**: TRL's `GRPOTrainer`, OpenRLHF's PPO trainer, verl's DAPO implementation. Read the actual training loop, line by line.
- **Hands-on**: Run GRPO on a 1.5B base model with TRL on GSM8K (math, ~\$30 of cloud compute). Then add a code-execution verifier and re-run on HumanEval.
- **Frontier**: agentic RL with tool-use, partial rollouts, multi-agent coordination, world models for agents.

### Track C — RLHF / Alignment (if Weeks 5–6 hit hardest)

The deepening path:
- **Nathan Lambert's RLHF book** — [rlhfbook.com](https://rlhfbook.com/) — the only end-to-end textbook on this topic. Read it.
- **Anthropic's Alignment Science** posts and **Alignment Forum** for the philosophical / safety side.
- **Hands-on**: train a reward model on UltraFeedback, then run DPO with TRL on a 1B SFT base. Compare your DPO-tuned model to the SFT base on Arena-Hard.
- **Frontier**: scalable oversight (debate, weak-to-strong generalization), interpretability for RM bias, Constitutional AI variants.

---

## 32. Hands-On Milestones

Pick three of these you haven't done and ship them in the next three months.

1. **Train DQN on CartPole** (Week 2 — `make train`) ✅ if you ran the demo.
2. **Train PPO on Atari Breakout** (Week 3 — `make train-breakout`). Watch the eval video — it learns to break bricks.
3. **Train DPO on a 350M–1B SFT model** with [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) using TRL. Should cost ~\$50 in cloud compute.
4. **Train GRPO on GSM8K with verifiable rewards** using TRL's `GRPOTrainer`. Should cost ~\$30. Demonstrates the R1 paradigm at small scale.
5. **Build a tiny agent RL environment** — wrap a search tool + a calculator, define a reward that incentivizes correct numerical answers, run GRPO.
6. **Reproduce the "ranking reverses with HP tuning" finding** from the Reddit thread (§30 #1). Run all three of PPO-RLHF, DPO, GRPO on the same model + dataset. Internalize the lesson.
7. **Read one paper a week from §35** for 8 weeks straight.

---

## 33. Open Research Questions

These are what 2026 papers are actively trying to answer. If you go to grad
school in this area, your thesis is somewhere in here.

- **How do you scale RLVR beyond verifiable domains?** Most useful tasks aren't math/code. What's the verifiable proxy for "good writing"?
- **Can RL-on-LLMs become sample-efficient enough for consumer hardware?** GRPO on a 1.5B model still needs 80GB. What breaks the bottleneck?
- **What's the right base for agent RL — pretrained, SFT'd, or post-RLHF?** R1-Zero's "no SFT" finding suggests pretrained-base RL might be optimal. But for agents?
- **Does reasoning RL learn transferable reasoning, or memorize templates?** The "faster, not smarter" finding (§21) says it might be the latter.
- **How do you align a model that's smarter than its labelers?** This is the "scalable oversight" problem and it's the central frontier alignment question.
- **What's the right reward function for multi-turn agentic tasks?** Single-turn reward (final answer correct) doesn't credit-assign across long tool-use chains.
- **Can reasoning RL replace SFT entirely?** If R1-Zero works, why do we need SFT at all? Answer is unclear and worth ~10 papers in the next two years.
- **Will RL hit a "data wall" like pretraining?** RLVR needs verifiable problems. Is there enough verifiable signal in the universe to train a frontier model?
- **What's the right RM architecture for long-context reasoning?** Current RMs are scalar-output classifiers; do we need step-level or rationale-aware RMs?
- **How does post-training compose?** SFT → DPO → RLVR → RLHF — what's the optimal order, and does it actually compose, or do later stages undo earlier ones?

---

## 34. Communities and Staying Sharp

### Subreddits worth following

The four mapped in §30 are the main ones; here's the broader landscape.

**Primary (highest signal for RL101 topics):**

- **r/reinforcementlearning** — best signal-to-noise for pure RL.
- **r/MachineLearning** — broader, but the long-form discussion threads are valuable.
- **r/LocalLLaMA** — model releases, deployment, fine-tuning gotchas. The "ground truth" of what's actually shipping.

**Secondary (worth checking weekly):**

- **r/learnmachinelearning** — entry-level RL/ML questions; useful when you're learning a new technique and want to see how others got stuck.
- **r/deeplearning** — broader deep-learning topics; some RL but mostly architectures and optimization.
- **r/huggingface** — TRL-specific issues, fine-tuning gotchas, dataset announcements.
- **r/singularity** — frontier-model news; high noise, but breaks paper releases fast.
- **r/AgenticAI** / **r/AIagents** — small but growing; agent-RL specific (Week 4 territory).

**Specialized:**

- **r/EleutherAI** — research-heavy, low-volume; great for open-source training discussion.
- **r/StableDiffusion** — different domain (image), but useful cross-pollination on RLHF/DPO applied to diffusion (DRaFT, DPOK, etc.).
- **r/MLQuestions** — Q&A format; sometimes finds answers to specific debugging issues faster than r/MachineLearning.
- **r/ChatGPTPro**, **r/ClaudeAI**, **r/OpenAI** — product-focused; training discussions surface occasionally.
- **r/Anthropic** — small, focused on Claude internals when employees post.

**Not worth following for RL101 topics**: r/artificial (very general), r/PromptEngineering (application-side, not training).

### Newsletters & blogs

- **Nathan Lambert — Interconnects** ([interconnects.ai](https://www.interconnects.ai/)) — the best for tracking the post-RLHF research front.
- **Sebastian Raschka — Magazine** ([magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/)) — monthly deep dives on training tricks.
- **Lilian Weng — Lil'Log** ([lilianweng.github.io](https://lilianweng.github.io/)) — encyclopedic explainers.
- **Anthropic Alignment Research** — the alignment-side counterpart.

### People to follow on X / Twitter

@karpathy, @NathanLambert, @sebastienbubeck, @hardmaru, @ylecun (sometimes spicy), @lmsysorg (Arena-Hard), @ctjlewis, @sidharthramachandran, @jxmnop.

### Conferences & venues

- **NeurIPS, ICML, ICLR** — the big three. NeurIPS in Dec, ICML in Jul, ICLR in May. Watch arXiv 2 weeks before submission deadlines.
- **RLC (Reinforcement Learning Conference)** — new dedicated RL venue, started 2024. Lower noise than NeurIPS for pure RL.
- **COLM (Conference on Language Modeling)** — language-model-specific venue, started 2024.

### Tracking new arxiv papers

The cs.LG and cs.CL arxiv listings are firehose-velocity — ~200 RL-and-LLM
papers per week. The trick is to use a curated layer.

**Best curated feeds (start here):**

- **HuggingFace Daily Papers** — [huggingface.co/papers](https://huggingface.co/papers). Community-voted, updated daily, most relevant for LLM-RL work. Good signal-to-noise.
- **alphaXiv** — [alphaxiv.org](https://www.alphaxiv.org/). Interactive arxiv with comments and trending feeds.
- **arxiv-sanity-lite** — [arxiv-sanity-lite.com](https://arxiv-sanity-lite.com/). Karpathy's project; lets you save custom keyword filters (e.g., "RLHF", "GRPO", "DPO", "RLVR") and get a personalized feed.
- **Papers with Code** — [paperswithcode.com](https://paperswithcode.com/). Trending papers with associated code; great for practical RL.

**Direct arxiv searches (bookmark these):**

- [arxiv RL+LLM recent (cs.LG)](https://arxiv.org/list/cs.LG/recent) — raw cs.LG listing.
- [arxiv search: RLHF OR GRPO OR DPO OR RLVR](https://arxiv.org/search/?searchtype=all&query=RLHF+OR+GRPO+OR+DPO+OR+RLVR&start=0) — saved keyword search.
- [arxiv cs.CL recent](https://arxiv.org/list/cs.CL/recent) — language-model papers.

**Live discussion:**

- **HuggingFace Discord** (very active, many paper authors lurking) and **EleutherAI Discord** (research-heavy) — papers usually get discussed within hours of release.
- Twitter/X researchers (see list above) — most papers are tweet-announced.

**A practical workflow** that doesn't burn you out:
1. Daily glance at HF Papers (5 min) — note titles only.
2. Weekly: skim arxiv-sanity-lite custom feed (15 min).
3. Monthly: read Lambert's *Interconnects* roundup (the curation work he does is worth its weight).
4. Quarterly: pick *one* paper that's been cited a lot, read it deeply (3-pass method below).

### How to read papers

First pass for the idea (intro + figures + conclusion); second pass for
the math (equations + experimental setup); third pass only if you're going
to implement or critique. The vast majority of arxiv papers don't deserve
the full third pass.

### Stay sharp without burning out

The RL/LLM literature is firehose-velocity. The trick is not to read
*everything* — pick one community or person whose taste you trust, and
read what they recommend. Lambert's Interconnects + Raschka's Magazine +
the curated arxiv-sanity list is enough.

---

# Part VI — Resources

---

## 35. Papers, Books, Videos, Code

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

## 36. Key Takeaways

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
11. **Algorithms differ in what they tolerate, not in absolute quality.** The Reddit pulse (§30) is right: PPO, DPO, GRPO are within hyperparameter tuning of each other. Pick the one whose failure modes you can debug.
12. **Chat-template hygiene matters more than algorithm choice.** The most common reason RLHF runs collapse is the policy and the reference disagreeing on prompt formatting. Always verify equality before debugging the loss.

---

## A final word

You started six weeks ago balancing a pole. You finish today knowing —
mathematically, architecturally, and culturally — how every model you
talk to gets trained.

The math, in the end, is small: a Bellman recursion, a clipped surrogate,
a Bradley-Terry log-sigmoid, a KL penalty. The hard part was never the
equations. It was knowing which hyperparameter to look at when the run
diverges, which framework to pick, which reward signal won't get hacked.

That's what you have now. Go build something with it.

---

*RL 101 Study Group — Colby Ziyu Wang @ SparkCraft / Hosted by AI Scholars*
*Notes: Alp Guneysel*

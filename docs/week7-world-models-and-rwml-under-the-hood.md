---
marp: true
theme: default
paginate: true
header: "RL 101 — Week 7 — World Models for LLM Agents (RWML)"
footer: "rl101-crash-course"
math: katex
---

<!--
To view as slides, install marp-cli and run:
  npx @marp-team/marp-cli docs/week7-world-models-and-rwml-under-the-hood.md --preview
  npx @marp-team/marp-cli docs/week7-world-models-and-rwml-under-the-hood.md --pdf
On GitHub this file also renders as a normal document with math and code.
-->

# World Models for LLM Agents — RWML Under the Hood

## Companion to RL 106 — World Models for LLM Agents

**Week 7 — RL 101 Study Group**

A slide-by-slide deep dive into Colby's RL 106 lecture, with a research
appendix on the model-based RL lineage that RWML descends from
(Dreamer V3, MuZero, V-JEPA 2, Genie 2, World Models), and a bridge
back to the agent-RL environments from Week 4 (ALFWorld, Tau²Bench)
and the RLVR shift from Week 6.

---

## How this doc relates to other weeks

- **Week 2 (DQN)** built up value-based RL and the Bellman view.
- **Week 3 (PPO)** introduced actor-critic and clipped policy gradient.
- **Week 4 (Forge)** set up the agent-RL environments — ALFWorld and
  Tau²Bench are the exact benchmarks the RWML paper reports on.
- **Week 5 (M2.7)** walked the RLHF stack with a production case study.
- **Week 6 (RLVR)** introduced "reality as verifier" — the math answer
  is the supervisor. RWML is the natural generalization: the *whole
  environment state* is the supervisor.
- **Week 7 (this doc)** asks: if reality can grade a math answer, can
  it teach an agent to *predict reality itself?*

If Week 6 was "the reward got harder to specify, so we replaced it with
a verifier," Week 7 is "the verifier is no longer the answer — it's the
*next observation*."

---

## Table of Contents

### Part I — Slide-by-Slide Companion (RL 106)

1. [Previously: RLHF Taught Preference — What's Missing](#1-previously-rlhf-taught-preference--whats-missing-slide-2)
2. [LLMs Can Talk, But Can They Predict Consequences?](#2-llms-can-talk-but-can-they-predict-consequences-slide-3)
3. [Biological Intelligence Imagines Before Acting](#3-biological-intelligence-imagines-before-acting-slide-4)
4. [What Is a World Model?](#4-what-is-a-world-model-slide-5)
5. [Why SFT Doesn't Build World Models](#5-why-sft-doesnt-build-world-models-slide-6)
6. [Introducing RWML](#6-introducing-rwml-slide-7)
7. [The Big Shift: Preference → Consequence](#7-the-big-shift-preference--consequence-slide-8)
8. [The RWML Pipeline](#8-the-rwml-pipeline-slide-9)
9. [The Environment Becomes the Teacher](#9-the-environment-becomes-the-teacher-slide-10)
10. [Semantic Reward, Not Token Match](#10-semantic-reward-not-token-match-slide-11)
11. [Why RL Instead of SFT?](#11-why-rl-instead-of-sft-slide-12)
12. [PPO Again — Except We Call It GRPO](#12-ppo-again--except-we-call-it-grpo-slide-13)
13. [Surprise Filtering — Skip the Easy Transitions](#13-surprise-filtering--skip-the-easy-transitions-slide-14)
14. [ALFWorld and Tau²Bench — The Two Examples](#14-alfworld-and-tau2bench--the-two-examples-slides-15-16)
15. [The Results — +19.6 and +7.9 Without Experts](#15-the-results--196-and-79-without-experts-slide-17)
16. [Catastrophic Forgetting and Why RWML Preserves Skills](#16-catastrophic-forgetting-and-why-rwml-preserves-skills-slide-18)
17. [Why World Models Matter](#17-why-world-models-matter-slide-19)
18. [Where AI Is Heading Next](#18-where-ai-is-heading-next-slide-20)
19. [Final Takeaways](#19-final-takeaways-slide-21)

### Part II — Research Appendix

20. [The Model-Based RL Lineage](#20-the-model-based-rl-lineage)
21. [Dreamer V3 — A World Model That Learned Minecraft](#21-dreamer-v3--a-world-model-that-learned-minecraft)
22. [MuZero — Planning Without Knowing the Rules](#22-muzero--planning-without-knowing-the-rules)
23. [V-JEPA 2 — LeCun's Predictive Architecture](#23-v-jepa-2--lecuns-predictive-architecture)
24. [Genie 2 — A Foundation World Model](#24-genie-2--a-foundation-world-model)
25. [How RWML Fits Into This Lineage](#25-how-rwml-fits-into-this-lineage)

### Part III — Bridge Back to Weeks 4–6

26. [The RLHF → RLVR → RWML Arc](#26-the-rlhf--rlvr--rwml-arc)
27. [ALFWorld and Tau²Bench Revisited (Week 4 Lens)](#27-alfworld-and-tau2bench-revisited-week-4-lens)
28. [GRPO in Week 4, Week 6, and Now Week 7](#28-grpo-in-week-4-week-6-and-now-week-7)

### Part IV — Q&A

29. [Q&A — RWML Specifics](#29-qa--rwml-specifics)
30. [Q&A — World Models vs. RLHF / RLVR](#30-qa--world-models-vs-rlhf--rlvr)
31. [Q&A — Practical and Open Questions](#31-qa--practical-and-open-questions)

### Part V — Where to Go from Here

32. [Hands-On Starting Points](#32-hands-on-starting-points)
33. [Open Research Questions](#33-open-research-questions)

### Part VI — Resources

34. [Papers, Videos, Code](#34-papers-videos-code)
35. [Key Takeaways](#35-key-takeaways)

---

# Part I — Slide-by-Slide Companion (RL 106)

---

## 1. Previously: RLHF Taught Preference — What's Missing (Slide 2)

The opening framing of the lecture is a deliberate handoff from Week 6.
What RLHF taught us in the previous six weeks:

- Preference learning (Bradley-Terry)
- Reward modeling
- PPO / GRPO / DPO for alignment

What it *did not* teach the model: an **internal world model**. A model
that has been RLHF'd knows what humans like to hear; it does not
necessarily know what happens next in the world.

The contrast the slide draws:

> RLHF teaches: *"What humans like."*
> World Models teach: *"What reality does."*

The dog analogy on this slide is sharper than it looks. A dog learns
from very few interactions because it has a built-in physics-and-causality
prior — it expects the world to behave in lawful ways. LLMs trained on
text don't have that prior automatically. They have a *language* prior
("here's what fluent humans usually write next"), which is structurally
different from a *world* prior ("here's what happens if I open this
drawer").

Whether this is an architecture problem or a training-data problem is
one of the live debates in the field (see §23 on V-JEPA). RWML takes the
training-data side of the bet: keep the architecture, change the data
distribution and the loss.

---

## 2. LLMs Can Talk, But Can They Predict Consequences? (Slide 3)

The slide names four things LLMs are already strong at:

- Question answering
- Code generation
- Multi-step reasoning
- Conversational fluency

And four things they still struggle with in *agentic* settings:

- Anticipating action consequences
- Adapting to environment dynamics
- Understanding hidden causal structure
- Long-horizon interaction

The thesis is direct: **agentic intelligence requires world
understanding, not just language understanding.** Good agents
don't only generate actions — they anticipate what happens after.

This is the exact failure mode Week 4 (Forge) wrestled with at the
systems level. The reason agent RL needs 100K+ environments and minutes-
long episodes is that learning "what happens when I call this tool" is a
*lot* harder than learning "what should I say next." Week 7 is the
algorithmic complement to Week 4's systems story.

---

## 3. Biological Intelligence Imagines Before Acting (Slide 4)

The dog/human framing here is the lecture's central intuition:

> Intelligence is not just reacting. It is predicting.

Three behaviors humans and animals do:

1. **Simulate outcomes** — "what happens if I press this?"
2. **Anticipate consequences** — "if I pick up the knife, the cutting board
   is still there."
3. **Mentally model environments** — "the kitchen is a place where these
   kinds of objects show up."

This is the **forward model** idea from neuroscience and classical
control theory. The brain does not just react to sensory input; it
*predicts* what its sensory input should be given its motor plan, and
the *prediction error* is what drives learning. Helmholtz called this
"unconscious inference" in the 1860s; modern computational neuroscience
calls it predictive coding.

The bet of Week 7 is that LLM agents need this same mechanism. Not
"predict the next token of fluent English," but "predict the next
*state* of the environment given the action I just took."

---

## 4. What Is a World Model? (Slide 5)

The slide gives three definitions, one per audience:

| View | Definition |
|------|------------|
| Beginner | "Predict what happens next" |
| Engineer | "Learn transition dynamics" |
| Researcher | "Approximate the environment transition function" |

Formally, a world model is a learned approximation $\hat T$ of the
environment's transition function:

```math
\hat T_\phi(s_{t+1} \mid s_{\leq t}, a_t) \approx P(s_{t+1} \mid s_{\leq t}, a_t)
```

Where $s_{\leq t}$ is the *history* of states (or observations) up to
time $t$, $a_t$ is the action, and $s_{t+1}$ is the next state. The
"$\leq t$" matters — most interesting environments are not Markov in raw
observations; you need history.

Two design axes the slide doesn't surface but matter a lot:

1. **What space do you predict in?** Raw pixels (Dreamer V1/V2),
   latent embeddings (Dreamer V3, V-JEPA), tokens (RWML),
   or game states (MuZero's hidden states)?
2. **Are you predicting one step or many?** Single-step transition
   models are easy to train but compound errors. Multi-step "imagination
   rollouts" (Dreamer V3's signature move) are harder to train but more
   useful for planning.

RWML in the lecture sits in the *token-space, single-step* corner: the
agent predicts the next observation (described in language) given the
history and the action it just took. This is the simplest viable world
model for an LLM agent — and the lecture's point is that *simplest viable*
is already enough to get measurable gains.

---

## 5. Why SFT Doesn't Build World Models (Slide 6)

The slide's argument is sharp:

- SFT requires expert demonstrations + curated datasets + exact target outputs.
- The model learns to **repeat the expert's wording.**
- But repeating wording is not the same as understanding the world.

The "knife on the countertop" example:

- A human says: *"There may be a knife on the countertop."*
- SFT learns: repeat this sentence.
- World modeling should learn: **why** the knife is likely there.

This is a subtle but real distinction. SFT optimizes:

```math
\mathcal{L}_{\text{SFT}} = -\sum_t \log p_\theta(y_t \mid x, y_{\lt t})
```

Token-level cross-entropy on the exact reference. There's no reward for
*meaning*; there's only reward for *match*. Two semantically identical
predictions can have arbitrarily different SFT losses if they're worded
differently. (This is also why SFT models often hedge in suspiciously
specific ways — they've learned the *surface form* of cautious language,
not the cautiousness itself.)

> SFT teaches agents to **sound** intelligent.
> World models teach agents to **think ahead**.

---

## 6. Introducing RWML (Slide 7)

The lecture names the technique: **Reinforcement World Model Learning
(RWML)**. The pitch in one sentence:

> Teach agents through interaction, consequences, and reinforcement
> signals to actually understand how the world changes.

The recipe is deceptively simple:

1. Ask the agent to predict the next state.
2. Ask it to reason about consequences.
3. Reward it based on feedback from interaction with the environment.

Notice what's *not* in the recipe: no expert labels, no stronger teacher
model, no human annotators. The environment itself is the signal.

RWML is best read as a generalization of RLVR (Week 6 §21). RLVR replaces
the human-preference reward with a *verifier* — code that returns 0/1.
RWML replaces the verifier with the *environment itself* — the next
observation is the ground truth that the prediction is compared against.

---

## 7. The Big Shift: Preference → Consequence (Slide 8)

This is the lecture's headline framing:

| | Source of supervision | What's modeled |
|---|---|---|
| **RLHF** | Human preferences | *"What humans prefer"* |
| **RWML** | Environment dynamics | *"What reality does"* |

Three levels at which the shift matters:

1. **Statistical** — preferences are averaged subjective judgments;
   environment transitions are objective facts. Variance behaves differently.
2. **Economic** — preferences require expensive human labelers; environment
   transitions are free (you just *run* the environment).
3. **Philosophical** — preferences encode "what we want the model to be
   like"; consequences encode "what the world is like." Different alignment
   target.

The shift is the same direction as RLVR, just one more step further
along: preferences → verifier → environment.

---

## 8. The RWML Pipeline (Slide 9)

The five-step loop the slide diagrams:

```
rollout → collect transitions → predict next state → compare with reality → RL update
                                                              ↑
                                                    no expert label needed
```

In a bit more detail:

1. **Rollout**: run the agent in the environment. Collect trajectories
   $(s_0, a_0, s_1, a_1, \dots, s_T)$.
2. **Collect transitions**: break trajectories into transition triples
   $(s_{\leq t}, a_t, s_{t+1})$.
3. **Predict next state**: for each $(s_{\leq t}, a_t)$, ask the LLM
   agent to *predict* $\hat s_{t+1}$.
4. **Compare with reality**: score the prediction against the actual
   $s_{t+1}$ the environment produced.
5. **RL update**: use the score as reward, run a policy-gradient step
   (PPO / GRPO).

Steps 1–2 are *data collection*; step 3 is the *forward pass* of the
world model; step 4 is the *reward function*; step 5 is the *policy
update*. The whole loop is self-supervised in the sense that no human
labels enter the system after the initial rollout policy is in place.

The non-obvious bit is step 4 — *how* do you compare a textual
prediction against a textual ground-truth state? Slide 11 (§10) gets to
the answer: semantic reward, not exact-token match.

---

## 9. The Environment Becomes the Teacher (Slide 10)

This slide makes the data side of step 1–2 concrete. Take the kitchen
example:

```
Trajectory:
  go countertop → see knife → open drawer → see spoon → ...

Transitions:
  (history=[go countertop], action=see, next=knife)
  (history=[go countertop, see knife], action=open drawer, next=spoon)
  ...
```

Formally, the transition is $(s_{\leq t}, a_t, s_{t+1})$:

- $s_{\leq t}$: the full interaction history up to step $t$ (this is
  what makes the model non-Markov-aware).
- $a_t$: the action taken (a token or phrase like "open drawer").
- $s_{t+1}$: the next observation as the environment returned it.

The slide's punchline:

> No experts. No labels. No stronger LLMs. The environment itself
> generates the learning signal.

This is the most economically important property of the technique. Every
prior generation of LLM post-training has a bottleneck somewhere — human
labelers for RLHF, ground-truth answers for RLVR, expert demonstrations
for SFT. RWML's bottleneck is *just running the environment*, which is a
compute cost rather than a human-time cost. That changes the scaling
curve.

---

## 10. Semantic Reward, Not Token Match (Slide 11)

The slide's example:

- **Real state**: *"You see a knife on the countertop."*
- **Prediction 1**: *"There is a knife near the counter."* → **High reward**
- **Prediction 2**: *"The kitchen explodes."* → **Low reward**

The reward function rewards *meaning* — not exact wording. The first
prediction is essentially correct (slight rephrasing, same content); the
second is nonsensical.

How do you compute this in practice? The lecture doesn't get into
implementation, but the standard approaches are:

1. **Embedding similarity** — embed prediction and ground truth, score
   with cosine similarity.
2. **Token overlap with smoothing** — BLEU, ROUGE, or chrF style metrics.
3. **LLM-as-judge** — a separate LLM grades "did the prediction capture
   what actually happened?" on a Likert scale or pairwise.
4. **Entailment-style** — does the prediction entail / get entailed by
   the ground truth?

In practice production systems usually combine 2 and 3: token-overlap
for cheap pre-filtering, LLM-judge for the final reward.

The deeper reason this works: the policy-gradient loss only ever uses
*reward differences*, not absolute rewards. As long as more-correct
predictions consistently score higher than less-correct ones, the
specific metric doesn't matter much. This is the same lesson Bradley-
Terry teaches in Week 6 (§9) — relative utility is enough.

---

## 11. Why RL Instead of SFT? (Slide 12)

The slide's three-tier framing again:

| Level | RL vs. SFT |
|------|------------|
| Beginner | SFT says "repeat exactly." RL says "figure out what works." |
| Engineer | RL optimizes semantic correctness; SFT optimizes token matching. |
| Researcher | SFT causes token-level collapse; RL allows multiple valid reasoning paths with the same semantic outcome. |

The researcher-level claim deserves unpacking. With SFT, two semantically
equivalent predictions with different wording get different losses; the
model has to *guess which exact wording is in the reference*. With RL on
a semantic reward, two semantically equivalent predictions get
approximately the *same reward*, and the gradient updates push the model
toward the *region* of correct meanings rather than the *point* of exact
wording.

> SFT: memorizing exam answers.
> RL: actually understanding the material.

This is the same intuition Week 6 §16 made for RLHF vs. SFT in general,
re-cast for world modeling specifically. The pattern is the same:
whenever you want to optimize for *meaning*, you need a reward signal
that respects semantic equivalence, and SFT does not.

---

## 12. PPO Again — Except We Call It GRPO (Slide 13)

This slide is a victory lap for Week 6 §20 and Week 4. The lecture
points out: the policy-update step of RWML is **GRPO**.

The setup:

- Sample $K$ predictions $\hat s^{(1)}_{t+1}, \dots, \hat s^{(K)}_{t+1}$
  for each transition.
- Score each one against the ground truth $s_{t+1}$ to get rewards
  $r_1, \dots, r_K$.
- Compute the group-relative advantage:

```math
A_i = \frac{r_i - \mathrm{mean}(r_{1..K})}{\mathrm{std}(r_{1..K})}
```

- Use the standard PPO clipped surrogate but with this advantage and *no
  critic*.

The "no critic" part is what makes this GRPO and not vanilla PPO.
There's no value network estimating $V(s)$; the baseline is just the
mean reward over the $K$ samples. The lecture's framing is exactly
right: GRPO replaces critics with *comparisons*. Actor-critic grades
your homework against an estimate of your average performance; GRPO has
students compare answers to each other.

> Actor-Critic: teacher grades your homework.
> GRPO: students compare answers with each other.

If you have the Week 6 §20 mental model, RWML's update step costs you
zero new conceptual machinery. The only new piece is the *reward
function* (semantic similarity to next-state ground truth) — the
*algorithm* is the same one DeepSeek-R1 used for math reasoning.

---

## 13. Surprise Filtering — Skip the Easy Transitions (Slide 14)

This is the most operationally important slide in the deck. Most
transitions in a kitchen environment are **boring**:

- "go to kitchen" → "you are in the kitchen"
- "look at table" → "you see a table"

The agent learns these in two rollouts and then gets *zero signal* from
further training on them. Training on easy transitions:

- Wastes compute
- Teaches nothing new
- Weakens the learning signal (lots of high-reward predictions drowns
  out the few hard ones)

The RWML solution: **filter for surprise**. Keep transitions where the
agent's prediction was *wrong* or *uncertain*; drop transitions it
already gets right.

Concretely, this is curriculum / importance-weighting. There are a few
ways to implement it:

1. **Reward-based filtering**: drop transitions where the average reward
   across the $K$ samples is above a threshold.
2. **Uncertainty-based filtering**: drop transitions where the model's
   prediction distribution has low entropy.
3. **Disagreement-based filtering**: drop transitions where all $K$
   samples agree (no group-relative variance, no gradient anyway).

DAPO (Week 6 §20) does the same thing for math reasoning — its "dynamic
sampling" trick re-samples prompts where every response got the same
reward, because zero-advantage prompts contribute zero gradient. RWML's
surprise filter is the world-model analog.

> Intelligence grows faster when learning focuses on difficult situations.

This is also the same intuition behind active learning, hard-example
mining, and prioritized experience replay in classic DQN. The principle
keeps showing up because it's true: the gradient comes from where the
model is wrong, not where it's already right.

---

## 14. ALFWorld and Tau²Bench — The Two Examples (Slides 15–16)

The lecture uses two specific benchmark environments:

### ALFWorld

A text-based household-task environment. The agent gets a goal like
"put a clean mug in the cabinet" and has to navigate a virtual house,
manipulate objects, and complete the task. The action space is natural
language commands ("go to kitchen", "open fridge", "pick up mug"); the
observation space is natural-language descriptions of what the agent
sees.

ALFWorld came up in Week 4 as one of the canonical agent-RL benchmarks.
What's different in Week 7: instead of training the agent on *task
success* (did you complete the goal?), you train it on *state
prediction* (given the action you just took, what does the next
observation look like?). The agent learns the *dynamics* of the
environment, not the policy directly.

### Tau²Bench (τ²-Bench)

A customer-service / tool-use benchmark. The agent plays a customer
service representative; it has access to APIs (lookup user info, modify
booking, etc.) and has to satisfy a customer goal across multiple
turns. The environment provides realistic tool responses and customer
follow-ups.

Tau²Bench is harder than ALFWorld in one important way: the customer's
*next utterance* is part of the next state, and that next utterance
depends on what the customer is feeling, what they want, and what the
agent just did. The world model has to predict not just "the API
returned X" but also "the customer's mood shifted because of how I
phrased the response." That's a richer, more social prediction target.

These two examples weren't chosen by accident. ALFWorld is the *easy*
case (deterministic, fully observable text world); Tau²Bench is the
*harder* case (partial observability, social dynamics, tool side
effects). Showing gains on both is the steel-man argument for RWML.

---

## 15. The Results — +19.6 and +7.9 Without Experts (Slide 17)

The headline numbers:

| Benchmark | RWML improvement |
|-----------|------------------|
| ALFWorld | **+19.6** |
| Tau²Bench | **+7.9** |

And the killer-app properties — these gains came **without**:

- ❌ Expert demonstrations
- ❌ Stronger teacher LLMs
- ❌ Task-success reward signals
- ❌ Human annotations

Let me unpack why each of these matters:

- **No expert demos** means the technique doesn't depend on having a
  good agent already; you can bootstrap from a weak base model.
- **No stronger teacher LLM** means there's no "distill from GPT-4"
  story hiding under the hood — the gains aren't from secretly using a
  bigger model.
- **No task-success rewards** means you don't need to design a reward
  function around the goal. The environment's *state* is the reward
  signal.
- **No human annotations** means the cost scales with compute, not with
  labeler hours. This is the same economic shift RLVR brought to math
  reasoning.

The agent learned from "interaction + consequences" alone. The lecture's
phrasing — that the agent "touched grass in the environment" — is funny
but precise. The world itself was the supervisor.

A caveat the lecture doesn't dwell on: +19.6 and +7.9 are *deltas* over
a base policy, not absolute scores. The base matters a lot. RWML
amplifies what the base model could partially do; it doesn't conjure
capability out of nowhere. (This is the same "faster, not smarter"
caveat from Week 6 §21 about RLVR on base models.)

---

## 16. Catastrophic Forgetting and Why RWML Preserves Skills (Slide 18)

The slide's framing:

> If humans study chess for 10 months, we usually do not suddenly forget
> calculus. But LLMs often do.

After aggressive SFT:

- Math ability drops
- Coding ability drops
- Reasoning quality drops

This is **catastrophic forgetting**, a well-known phenomenon since
McCloskey & Cohen (1989). The cause in modern LLMs:

- SFT does *token-level* updates on a narrow distribution.
- Large gradients flow through the same parameters that store unrelated
  capabilities.
- Other capabilities get overwritten as side effects.

The slide's claim: **RWML preserves prior capabilities better than SFT.**
Why?

1. **Reward-weighted updates instead of likelihood updates.** RL updates
   only flow when the reward signal disagrees with current behavior.
   On capabilities the model already has correct (math, code, reasoning),
   the reward agrees and the gradient is small.
2. **Diverse exploration.** Sampling $K$ predictions per transition
   keeps the policy from collapsing to a narrow mode.
3. **KL anchor.** Standard PPO/GRPO uses a KL penalty to the reference
   model, which acts as a regularizer against drifting too far from
   the base distribution. (This is the same KL trick from Week 6 §12.)

The practical implication is huge: RWML lets you *add* world-modeling
capability without *subtracting* general capability. SFT typically
forces a trade-off; RWML doesn't.

Empirically the evidence for "RL preserves capability better than SFT"
is well-supported (see the InstructGPT paper, the R1-Zero results, and
the Tülu 3 paper from Week 6). The Week 7 contribution is showing it
applies to world-model learning specifically, not just preference
alignment.

---

## 17. Why World Models Matter (Slide 19)

The slide lists four downstream uses for a world model:

| Capability | What it enables |
|------------|-----------------|
| Planning | "What should I do next?" |
| Anticipation | "What might happen if I act?" |
| Efficient exploration | "Which actions are worth trying?" |
| Long-horizon reasoning | "How will this affect the future?" |

The two-line summary:

- Without world models: the agent is speedrunning life blindly.
- With world models: the agent mentally simulates outcomes before acting.

The most important of the four is **planning**. With a world model in
hand, you can use *Monte Carlo Tree Search* (or rollout-based planning,
or model-predictive control) at decision time: simulate possible action
sequences, score them with the world model, pick the best one. This is
what AlphaGo, AlphaZero, and MuZero do, and it's how planning got to
superhuman levels in board games.

In LLM-agent terms, this is what would let an agent say "if I send this
API call, the user will probably push back, then I'd need to escalate,
so let me start with a different approach" — actual chain-of-consequence
reasoning, not just chain-of-thought verbalization.

The lecture's framing:

> Prediction is a core ingredient of intelligence.

That's the bet of the whole world-models research program, from Ha &
Schmidhuber's original *World Models* paper (2018) to LeCun's V-JEPA
manifesto.

---

## 18. Where AI Is Heading Next (Slide 20)

The lecture frames the trajectory:

```
Chatbots → Agents → World Models → Autonomous Systems
```

Future LLM agents will need:

- Memory (long-horizon state)
- Planning (search over future actions)
- Simulation (predict consequences)
- Environment understanding (causal models)
- Long-term decision making (credit assignment across many steps)

The summary:

> The future of AI is not just conversation. It is interaction with reality.

This framing is widely shared but not universal. The dissenters' position
(see V-JEPA / LeCun for the most-developed version): pure token-prediction
LLMs *cannot* learn world models well enough, full stop. World models
need a different architecture — predictive embedding spaces, joint-
embedding predictive architectures, or fundamentally non-autoregressive
designs.

RWML takes the *opposite* bet: the architecture is fine, the loss is the
problem. Both bets are live research programs.

---

## 19. Final Takeaways (Slide 21)

The lecture's five-point summary:

1. **RLHF teaches: human preference.**
2. **RWML teaches: environment dynamics.**
3. **World models improve: planning and decision-making.**
4. **Self-supervised RL can teach: powerful agent behavior without experts.**
5. **The big philosophical shift: from predicting tokens → to predicting consequences.**

And the closing line, which is worth taking seriously:

> The future of AI is not just generating text — it is understanding reality.

The strong-form version: prediction *is* intelligence. The weak-form
version: prediction is *necessary* for intelligence but not sufficient.
Most of the field is currently betting somewhere in between.

---

# Part II — Research Appendix

The lecture introduces RWML as if it sprang out of nowhere, but it sits
at the end of a long lineage of model-based RL research. The taxonomy
matters because it tells you what's actually new (a few specific
choices) and what's borrowed (most of the machinery).

---

## 20. The Model-Based RL Lineage

Model-based RL has three eras, very roughly:

### Era 1 — Classical world models (1990–2010s)

- Dyna-Q (Sutton, 1990) — interleave real and imagined transitions in
  Q-learning.
- PILCO (Deisenroth & Rasmussen, 2011) — Gaussian-process world models
  for continuous control.
- World Models (Ha & Schmidhuber, 2018) — first deep world model
  scaled to pixel-input environments; LSTM-over-VAE-latents architecture.

### Era 2 — Deep world models for games and control (2018–2024)

- Dreamer V1 (Hafner et al., 2019) — RSSM (Recurrent State-Space Model)
  + latent imagination.
- Dreamer V2 (Hafner et al., 2020) — discrete latents; Atari at
  human-level performance from images alone.
- Dreamer V3 (Hafner et al., 2023) — *learned Minecraft diamond from
  scratch*, the first world model to do so.
- MuZero (Schrittwieser et al., 2019) — learned dynamics + MCTS, mastered
  Atari/Go/Chess/Shogi without being told the rules.

### Era 3 — World models for LLM agents (2024–now)

- V-JEPA / V-JEPA 2 (Meta / LeCun, 2024–2025) — predictive embedding
  architecture for video.
- Genie 2 (DeepMind, 2024) — foundation world model for 3D
  interactive environments.
- **RWML** (this lecture, ~2025–2026) — bring the world-model idea to
  LLM agents specifically.

The Week 7 contribution is best understood as the *third era's
LLM-specific instance*: take the world-model idea, simplify the
architecture down to "predict the next observation token-by-token," and
plug it into the GRPO machinery that Week 6 already built.

---

## 21. Dreamer V3 — A World Model That Learned Minecraft

The Dreamer line (Hafner et al.) is the closest classical analog to RWML
in spirit, though very different in architecture.

**Architecture** — Recurrent State-Space Model (RSSM):

- An encoder turns raw observation into a latent $z_t$.
- A recurrent model maintains a deterministic history state $h_t$.
- The world model predicts $z_{t+1}$ given $(h_t, z_t, a_t)$.
- A separate "reward head" predicts the reward for each latent state.
- A policy and value are trained *purely on imagined rollouts* through
  the world model — no real environment steps for the policy gradient.

**The Minecraft result (Hafner et al., 2023)**:

- Dreamer V3 became the first agent to mine a diamond in Minecraft from
  scratch — no human play, no expert demos, no curriculum.
- This was significant because diamond requires ~24 sub-goals (mine
  wood, craft pickaxe, mine stone, etc.) and previous agents had needed
  hand-engineered curricula or imitation learning.

**What Dreamer shares with RWML**:

- Train a forward dynamics model from interaction.
- Use that model to improve the policy.
- No expert labels — the environment is the signal.

**What's different**:

- Dreamer predicts in *latent space*, not text. The latents are
  abstract floating-point embeddings, not natural-language descriptions.
- Dreamer's policy is trained on *imagined* rollouts inside the world
  model. RWML's policy is trained on *real* rollouts with the world
  model as a reward signal.
- Dreamer is a *complete* RL system (policy + value + world model).
  RWML is the world-model-learning step plugged into existing LLM-agent
  training.

The Dreamer V3 paper is probably the best deep technical companion to
this lecture for anyone with an RL background.

---

## 22. MuZero — Planning Without Knowing the Rules

MuZero (Schrittwieser et al., DeepMind, 2019; published Nature 2020) is
the spiritual predecessor of all "predict the future to plan" methods.

**The trick**:

- Instead of learning the *true* transition function $P(s'|s,a)$, learn
  an *abstract* one. Define a hidden state $h_t$, a dynamics function
  $g(h_t, a_t) \to (h_{t+1}, r_{t+1})$, a policy head $\pi(h_t)$, and a
  value head $V(h_t)$.
- Train $g$ such that the *predicted reward and value* match the real
  ones — even if the hidden state $h_t$ has no interpretable meaning.

**The result**:

- Matched AlphaZero on Go, chess, and shogi *without being told the
  rules of the game*.
- Matched the state-of-the-art on Atari with the same machinery.

**The lesson MuZero teaches RWML**:

- The world model doesn't have to predict *raw* observations to be useful
  for planning. It just has to predict *task-relevant* features.
- You can train the world model end-to-end with the policy, jointly.

RWML doesn't use planning at decision time (it's pure policy-gradient,
no MCTS). But the door is open: once you have a world model that
predicts next states well enough, you can *bolt on* MCTS-style planning
at inference. That's the natural next step for the technique.

---

## 23. V-JEPA 2 — LeCun's Predictive Architecture

Yann LeCun has been arguing for years that next-token prediction is the
wrong objective for world modeling. The **JEPA** (Joint-Embedding
Predictive Architecture) family is his alternative.

**The architecture**:

- Encode the current observation into an embedding $s_x$.
- Encode the future observation into an embedding $s_y$.
- A predictor $P$ tries to map $s_x$ to $s_y$ in embedding space.
- The loss is similarity between $P(s_x)$ and $s_y$, *not* reconstruction
  of pixels.

**Why predict in embedding space, not pixel space**:

- Pixel-level prediction wastes capacity on irrelevant details (the
  texture of every leaf in a forest).
- Embedding-level prediction lets the encoder choose what's worth
  preserving.

**V-JEPA 2 (2024)** scaled this to video and showed strong zero-shot
transfer on action recognition and physical-reasoning tasks.

**LeCun's claim**: this — *not* autoregressive token prediction — is
the right architecture for world models. The autoregressive bet (which
RWML implicitly makes) is, in his view, structurally limited.

**RWML's counter-bet**: the architecture is fine; you just need the
right loss. Predict tokens, but reward semantic correctness against
real environment outcomes.

Both bets are live. V-JEPA-style architectures are showing up in
robotics and embodied AI; autoregressive world-modeling is showing up
in agent RL like RWML. Neither has decisively won yet.

---

## 24. Genie 2 — A Foundation World Model

Genie 2 (DeepMind, Dec 2024) is the *generative* counterpart to all of
the above. Given a single image and a sequence of keypresses, Genie 2
generates an interactive video — you can "play" in a hallucinated 3D
environment that was never explicitly modeled.

**Why this matters for RWML**:

- Demonstrates that world models can scale to general, open-domain
  environments — not just curated benchmarks.
- Demonstrates that a foundation-model-style training recipe works for
  world models (pretrain on tons of video, fine-tune on specific tasks).
- Suggests a path where the *base* of an LLM-agent system is a
  pretrained world model, and RWML fine-tunes it for specific
  environments. This is the *world-model pretraining* future the lecture
  hints at but doesn't develop.

Genie 2 generates pixels, not text, so it's not directly applicable to
ALFWorld / Tau²Bench. But the architectural lesson — that you can
*pretrain* a world model the same way you pretrain a language model —
is the bridge that connects the world-model-RL literature to the
foundation-model literature.

---

## 25. How RWML Fits Into This Lineage

Quick reference map:

| Method | Year | Predicts in | Loss signal | Domain |
|--------|------|-------------|-------------|--------|
| Dyna-Q | 1990 | State space | Bellman target | Tabular RL |
| World Models (Ha & S.) | 2018 | Latent (VAE) | Reconstruction | Pixel envs |
| MuZero | 2019 | Hidden state | Reward + value | Board games, Atari |
| Dreamer V3 | 2023 | Latent (RSSM) | Reconstruction + reward | Atari, Minecraft |
| V-JEPA 2 | 2024 | Embedding | Embedding similarity | Video |
| Genie 2 | 2024 | Pixels | Generative loss | Open-domain video |
| **RWML** | 2025-ish | **Tokens (text)** | **Semantic reward** | **LLM agents** |

What RWML brings to the table that's distinctively new:

- **Token-space prediction** — the world model is just the LLM, with no
  separate encoder/decoder.
- **Semantic reward, not reconstruction loss** — you don't need
  pixel-level fidelity; you need meaning-level correctness.
- **GRPO as the policy update** — group-relative comparison, no critic.
- **Direct compatibility with LLM-agent infrastructure** — you can run
  this on top of any existing RLHF/RLVR pipeline.

What RWML borrows:

- The world-model idea itself (Ha & Schmidhuber, Dreamer).
- Learning from interaction without expert labels (model-based RL since
  forever).
- Curriculum / surprise filtering (active learning, hard-example mining).
- GRPO (DeepSeek, Week 6 §20).

---

# Part III — Bridge Back to Weeks 4–6

---

## 26. The RLHF → RLVR → RWML Arc

The clearest way to see Week 7 is as the third step in a sequence that
spans Weeks 5–7:

| Era | Reward source | What it scales |
|-----|---------------|----------------|
| **RLHF** (Week 5–6) | Human preferences via RM | Subjective alignment ("be helpful") |
| **RLVR** (Week 6 §21) | Verifier (rule-based / unit test) | Verifiable correctness ("get the math right") |
| **RWML** (Week 7) | Environment state | Dynamics understanding ("predict what happens next") |

Each step:

1. Removes a labeling bottleneck (humans → code → environment).
2. Expands the *domain* of training (alignment → math/code → agentic
   tasks).
3. Keeps the same core algorithm (PPO / GRPO).

The Week 6 synthesis line — *the reward source moved: human → RM → DPO
(no RM) → verifier → environment* — completes here. RWML is the
"environment" endpoint.

A related arc on the *base model* side:

- **RLHF** assumes you start from an SFT-tuned chat model.
- **RLVR** can work from a pretrained base (R1-Zero showed this).
- **RWML** works from any base that can role-play an agent — the gain
  in §15 is on top of an already-trained agent base.

The base keeps getting cheaper / more permissive as the supervisor
signal gets richer.

---

## 27. ALFWorld and Tau²Bench Revisited (Week 4 Lens)

Week 4 introduced ALFWorld and Tau²Bench as the canonical agent-RL
benchmarks — environments where MiniMax Forge stress-tested its
PPO → GRPO → DAPO → CISPO algorithmic stack.

Week 7 uses the *same environments* but trains on a *different signal*:

| | Week 4 (Forge) | Week 7 (RWML) |
|---|---|---|
| What's the reward? | Task success (did the agent complete the goal?) | Next-state prediction accuracy |
| What does the policy optimize? | Action selection | State *understanding* |
| What does success look like? | Higher task completion rate | Higher next-state prediction quality |

These are complementary, not competing. A real production agent could
use *both* — RWML for world-model warm-up, then task-success RL for
final policy optimization. The lecture hints at this without naming it
explicitly.

The other Week 4 connection: prefix-tree training (Magi Attention) and
partial rollouts are exactly the systems infrastructure RWML would need
to scale. The trajectory storage cost of "every transition is a training
example" is non-trivial; Forge's prefix-tree memory layout is
state-of-the-art for that.

---

## 28. GRPO in Week 4, Week 6, and Now Week 7

GRPO has now shown up three times in the course:

| Week | Where GRPO appears | Reward signal |
|------|--------------------|---------------|
| Week 4 (Forge) | Algorithmic lineage, MiniMax M1 used CISPO (GRPO descendant) | Task success on agent benchmarks |
| Week 6 (RLVR) | DeepSeek-R1's primary algorithm | Verifier (math correctness, code unit tests) |
| Week 7 (RWML) | The update step of RWML | Semantic similarity to next environment state |

The algorithm doesn't change. The *reward function* changes. This is the
single most important meta-lesson of Weeks 4–7: **RL on LLMs is
increasingly about reward engineering, not algorithm engineering.** Once
you have GRPO + a stable PPO clip + a KL anchor, the entire game is
"what reward signal will scale to my domain?"

If you only remember one thing from Weeks 5–7, make it this. Algorithm
choice matters at the margin; reward design matters at the order of
magnitude.

---

# Part IV — Q&A

---

## 29. Q&A — RWML Specifics

**Q1. Why predict the next state in token space instead of in an embedding space like Dreamer V3?**
Three reasons. (1) The LLM is already a token-prediction model — no extra encoder/decoder needed. (2) Tokens are interpretable; you can read the prediction and debug it. (3) Semantic reward works in token space because LLM judges and embedding similarity both handle natural language directly. The trade-off is efficiency — embedding-space prediction is cheaper because the prediction target is lower-dimensional.

**Q2. What's the actual reward function used for "did the prediction match reality?"**
The lecture doesn't fully specify, and there's no single answer in the literature. The standard implementations combine: (a) cheap pre-filtering with chrF or BLEU-style token overlap, (b) LLM-as-judge for the final score, (c) optionally an embedding-similarity term to handle paraphrasing. The exact recipe is one of the unspecified details that probably matters a lot in practice.

**Q3. How is "surprise filtering" different from active learning?**
It's not, conceptually. Surprise filtering is active learning re-branded for the RL setting. The novelty is the *application* — using it to drop boring transitions in an agent-RL trajectory store — not the mechanism. The same idea shows up in prioritized experience replay (DQN), hard-example mining (object detection), and DAPO's dynamic sampling (Week 6).

**Q4. Why does the agent improve on ALFWorld and Tau²Bench when the reward is "predict the next state" and not "complete the task"?**
The deep answer: predicting the next state requires *understanding the environment dynamics*, which is a prerequisite for completing the task. A model that can predict "if I open the drawer, I will see a spoon" is implicitly modeling the kitchen layout — and that implicit model helps it choose better actions even when the explicit reward is just prediction accuracy. This is called *representation transfer* — the world model becomes a useful internal abstraction even for downstream tasks.

**Q5. Does RWML need rollouts in the actual environment, or can it work on logged data?**
The lecture assumes online rollouts. In principle you could do offline RWML on logged trajectories (you have $(s_{\leq t}, a_t, s_{t+1})$ tuples either way). But offline RL is notoriously harder — you can't sample $K$ predictions and *test them against new ground truth*, only against the one ground truth you have. Group-relative advantage degenerates to absolute reward.

---

## 30. Q&A — World Models vs. RLHF / RLVR

**Q6. Is RWML a replacement for RLHF or RLVR, or a complement?**
A complement. RLHF optimizes for human preferences (alignment); RLVR optimizes for verifiable correctness (reasoning); RWML optimizes for environment understanding (world modeling). They target different capabilities. A production agent in 2026+ likely uses all three at different stages.

**Q7. What's the relationship to predictive coding in neuroscience?**
RWML is structurally close to predictive coding (Rao & Ballard, 1999; Friston's free-energy principle): the brain predicts its sensory input and learns from prediction error. RWML predicts the next observation and learns from the error. The disanalogies are mostly about scale and implementation (LLMs don't have the hierarchical predictive structure neuroscience theorizes), but the loss signal is the same.

**Q8. Won't this fail catastrophically on stochastic environments?**
Yes, it would fail on *adversarially* stochastic environments (e.g., flip a fair coin every step) — the world model can't predict the coin and gets zero signal from it. But for *most* real environments, transitions are mostly deterministic conditioned on enough history; the residual stochasticity is bounded. Tau²Bench has stochastic customer behavior and RWML still wins, so the stochasticity tolerance is at least non-trivial.

**Q9. How does RWML interact with RAG / tool use?**
Open question. If the agent calls a tool (search the web, query a DB), the "next state" includes the tool's response, which is a function of an external system the world model can't fully internalize. The reasonable interpretation: the agent learns to predict *what tools tend to return* in different contexts (a learned API model), not the underlying world. This is a useful capability but probably easier than full world modeling.

**Q10. Is the LLM itself the world model, or is there a separate world-model network?**
The lecture's RWML uses the LLM itself — the same model that selects actions also predicts next states. This is parameter-efficient but invites entanglement (the policy and the world model share weights and can interfere). Separate networks are possible and would mirror the actor-critic split, but at extra cost.

---

## 31. Q&A — Practical and Open Questions

**Q11. What's the smallest model I can do RWML on?**
Speculation, not measured: probably 1–3B for ALFWorld-scale environments. RWML is easier than full agent-RL because the loss is denser (every transition is a training signal, not just task completion). You don't need 7B or above to see signal; you need enough capacity to model the environment's dynamics, which for text worlds isn't enormous.

**Q12. How do you avoid the model just memorizing the training trajectories?**
Same way RL handles it in general: KL anchor to the reference model, sampling temperature, surprise filtering (skip transitions the model already gets right). The danger is most acute in deterministic environments where memorization is feasible; in stochastic ones, memorization is a weaker attractor because the "right answer" is itself uncertain.

**Q13. Does RWML scale to long-horizon environments (10K+ steps)?**
Unclear and probably the biggest open question. The transition format $(s_{\leq t}, a_t, s_{t+1})$ has $s_{\leq t}$ growing linearly with $t$; at 10K steps this is a 10K-token history per transition. Prefix-tree training (Week 4) helps amortize that cost, but the actual experimental evidence at long horizons is sparse.

**Q14. Can you bootstrap RWML from a pretrained world-model foundation (like Genie 2)?**
Architecturally plausible, empirically untested for text agents specifically. The analog would be: pretrain on a huge corpus of agent trajectories (synthetic or scraped), then RWML-fine-tune on the target environment. This is the natural "world-model foundation model" play — and probably the next major direction the field goes.

**Q15. What if the environment has hidden state the agent can't observe?**
Then the world model becomes a *predictive model of partial observations*, which is strictly harder but still tractable. POMDPs (partially-observed MDPs) have a 30-year theoretical literature; the practical answer is "give the agent enough history and hope." Tau²Bench's customer-mood-modeling is exactly this case — the customer's internal state isn't directly observable, but the agent can predict the customer's *next utterance* from enough history.

**Q16. How do you debug an RWML run that's diverging?**
Same general debugging recipe as any RL run (Week 6 §29 Q20), with these additions:
1. Plot prediction-quality reward over time — if it stops improving, your world model isn't learning.
2. Manually inspect $K$-sample predictions for a few hard transitions — if all samples are identical, your sampling temperature is too low or your KL anchor is too tight.
3. Sanity-check the semantic reward function — if it can't distinguish "knife on counter" from "kitchen explodes," fix the reward before the algorithm.
4. Apply surprise filtering — if 90% of transitions are getting maxed-out reward, you're training on noise.

**Q17. Is RWML "world modeling" in the strong sense, or just sophisticated next-token prediction?**
Honest answer: closer to the latter than the lecture admits. The model is still doing autoregressive next-token prediction; the change is the *training distribution* (state transitions instead of preference pairs) and the *reward function* (semantic similarity to environment ground truth instead of preference). It's not a *new* mechanism for understanding causality — it's a new place to apply the existing mechanism. Whether that counts as "real" world modeling depends on how strong a definition you take. The pragmatic answer: it works, and that's what matters.

---

# Part V — Where to Go from Here

---

## 32. Hands-On Starting Points

If you want to actually try RWML-style training:

1. **Read the Dreamer V3 paper first.** It's the cleanest example of
   model-based RL with a real implementation. *Mastering Diverse Domains
   through World Models* — Hafner et al., 2023, [arxiv:2301.04104](https://arxiv.org/abs/2301.04104).
2. **Set up ALFWorld locally.** It's a text-based env, runs on CPU, and
   is the canonical agent-RL benchmark. [github.com/alfworld/alfworld](https://github.com/alfworld/alfworld).
3. **Replicate the simplest possible RWML loop on ALFWorld**:
   - Run a base LLM agent on ALFWorld, log all transitions.
   - Train a separate small LLM to predict next observations from
     transitions.
   - Score the predictions with chrF + LLM-judge.
   - Run GRPO with that reward on the prediction model.
   - Measure: does the agent get better at ALFWorld when fine-tuned with
     this signal?
4. **Compare to a pure RLVR baseline** on the same environment. The
   question to answer for yourself: how much of the gain is "RL on agent
   trajectories" and how much is specifically "world modeling"?
5. **Read the original Ha & Schmidhuber World Models paper** ([2018, arxiv:1803.10122](https://arxiv.org/abs/1803.10122)). It's short and historically important.

---

## 33. Open Research Questions

These are the questions Week 7 leaves unanswered:

- **What's the right reward function for state prediction?** chrF? LLM
  judge? Embedding similarity? Some hybrid? The lecture doesn't say and
  the literature doesn't have consensus.
- **Does RWML scale to long-horizon environments?** Current results
  are on ALFWorld-scale (~tens of steps) and Tau²Bench (~tens of
  turns). What happens at 1K+ steps?
- **Can you pretrain a world-model foundation model for LLM agents?**
  Genie 2 did this for pixels. The text-agent analog would be
  pretraining on millions of agent trajectories.
- **Does the world model transfer across environments?** Or do you have
  to re-train it for every new environment?
- **What's the relationship between world-model quality and downstream
  task performance?** RWML shows correlation; the causal story is still
  open.
- **How does this interact with multi-agent settings?** When the
  "environment" includes other agents whose behavior you also have to
  predict, the world model has to model *their* world model. The
  recursion is non-trivial.
- **Can you do MCTS / planning at decision time using the world model?**
  MuZero does this for board games; nobody has yet for LLM agents at
  scale. This is probably the highest-impact open direction.
- **Is autoregressive next-token prediction structurally limited as a
  world model?** LeCun's bet says yes; RWML's results say not yet. The
  debate isn't settled.

---

# Part VI — Resources

---

## 34. Papers, Videos, Code

### Foundational world-model papers (read in order)

1. **Ha & Schmidhuber 2018** — *World Models* — [arxiv:1803.10122](https://arxiv.org/abs/1803.10122). The origin. Short, readable, historically important.
2. **Schrittwieser et al. 2019** — *MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* — [arxiv:1911.08265](https://arxiv.org/abs/1911.08265).
3. **Hafner et al. 2019** — *Dreamer V1: Dream to Control* — [arxiv:1912.01603](https://arxiv.org/abs/1912.01603).
4. **Hafner et al. 2020** — *Dreamer V2: Mastering Atari with Discrete World Models* — [arxiv:2010.02193](https://arxiv.org/abs/2010.02193).
5. **Hafner et al. 2023** — *Dreamer V3: Mastering Diverse Domains through World Models* — [arxiv:2301.04104](https://arxiv.org/abs/2301.04104). The Minecraft-diamond paper.

### Predictive-architecture line

6. **LeCun 2022** — *A Path Towards Autonomous Machine Intelligence* — [openreview](https://openreview.net/forum?id=BZ5a1r-kVsf). The JEPA manifesto.
7. **Bardes et al. 2024** — *V-JEPA 2* (Meta AI) — search "V-JEPA 2" on arxiv (preprint released 2024–2025). Predictive embedding architecture for video.

### Generative world models

8. **Bruce et al. 2024** — *Genie: Generative Interactive Environments* (DeepMind). The original Genie paper, [arxiv:2402.15391](https://arxiv.org/abs/2402.15391). Genie 2 is the follow-up (blog post, Dec 2024).

### LLM-agent environments

9. **Shridhar et al. 2021** — *ALFWorld: Aligning Text and Embodied Environments for Interactive Learning* — [arxiv:2010.03768](https://arxiv.org/abs/2010.03768).
10. **τ-Bench / τ²-Bench** — search "tau-bench" on arxiv. Customer-service tool-use benchmark from Sierra.

### Related Week 6 reading (RLVR / GRPO)

11. **DeepSeekMath / GRPO** — [arxiv:2402.03300](https://arxiv.org/abs/2402.03300).
12. **DeepSeek-R1** — [arxiv:2501.12948](https://arxiv.org/abs/2501.12948). RLVR + GRPO; the algorithmic backbone RWML reuses.

### Blogs and explainers

- **Danijar Hafner — Dreamer blog post** — [danijar.com/dreamer/](https://danijar.com/dreamer/). The Dreamer line is mostly his work; his blog is the best entry point.
- **Yann LeCun on world models** — search his recent talks; the *Autonomous Machine Intelligence* talks from 2022–2024 are the most-cited.
- **DeepMind Genie 2 announcement** — [deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/).

### Videos

- **Andrej Karpathy — Deep Dive into LLMs (2024)** — [youtube.com/watch?v=7xTGNNLPyMI](https://www.youtube.com/watch?v=7xTGNNLPyMI). Covers RLVR; sets up the world-model discussion in context.
- **AI Scholars — RL 101 Past Sessions** — [YouTube playlist](https://www.youtube.com/watch?v=4e0laDA7jlM&list=PLte0_KfXCwoh2EX7KRmooLU-Jyn-y8BQZ).

### Code

- **Dreamer V3 (official)** — [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3).
- **MuZero (open implementation)** — [github.com/werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general).
- **ALFWorld** — [github.com/alfworld/alfworld](https://github.com/alfworld/alfworld).
- **TRL (GRPO trainer)** — [github.com/huggingface/trl](https://github.com/huggingface/trl). The GRPO update used by RWML lives here.
- **verl** (ByteDance) — [github.com/volcengine/verl](https://github.com/volcengine/verl). DAPO/GRPO implementations.

### Companion docs in this repo

- **Week 4** — [`docs/week4-agent-rl-forge.md`](week4-agent-rl-forge.md) — ALFWorld / Tau²Bench environments, GRPO→DAPO→CISPO lineage.
- **Week 6** — [`docs/week6-rlhf-and-rft-under-the-hood.md`](week6-rlhf-and-rft-under-the-hood.md) — full GRPO derivation, RLVR explanation, the reward-source arc.

---

## 35. Key Takeaways

1. **RWML replaces the reward with the environment itself.** No experts, no stronger teachers, no human labels, no task-success signal — just "did your prediction match what happened next?"
2. **The pipeline is a five-step loop.** Rollout → collect transitions → predict next state → compare with reality → GRPO update. Each step is borrowed from existing machinery.
3. **The world model is the LLM itself.** No separate encoder, no latent space, no separate dynamics network. Just predict the next observation in token space.
4. **Semantic reward, not token match.** The reward function rewards meaning-preserving predictions. Implementation is open: chrF + LLM-judge is the common recipe.
5. **GRPO is the update step.** Group-relative advantage, no critic — same algorithm as DeepSeek-R1, repurposed for state prediction. Algorithm reuse across Weeks 4, 6, and 7 is the meta-lesson.
6. **Surprise filtering is the operational secret.** Drop transitions the agent already gets right; train on the hard ones. Same principle as DAPO's dynamic sampling and prioritized experience replay.
7. **ALFWorld +19.6 and Tau²Bench +7.9** without expert demos, stronger teachers, task-success signals, or human annotations.
8. **RWML preserves prior capabilities better than SFT.** Reward-weighted updates don't flow where the model is already correct, so old skills don't get overwritten.
9. **The reward source arc completes here.** Human → RM → DPO (no RM) → verifier (RLVR) → **environment (RWML)**. Each step removes a labeling bottleneck.
10. **RL on LLMs is now mostly reward engineering.** GRPO is stable; the design question is what *signal* you train on. Weeks 5–7 are three different answers to that one question.
11. **World models matter because prediction enables planning.** With a world model in hand, you can do MCTS-style search at decision time — the natural next step the lecture hints at but doesn't develop.
12. **The autoregressive bet is live.** LeCun argues next-token prediction is structurally limited for world modeling. RWML argues the architecture is fine; the loss is the problem. Both bets have results; neither has won.

---

## Looking ahead

Week 7 closes the trilogy that started in Week 5: preferences → verifier
→ environment. Week 8 (recommended-lecture slot — TBD) is the natural
follow-up: take the algorithms we've built and put them to work in a
*physical* simulator. MuJoCo or Isaac Sim, classical continuous control,
the same PPO/GRPO machinery applied to a different action space. The
through-line from CartPole in Week 2 to humanoid locomotion in Week 8
is the same algorithm — just better simulators and harder rewards.

---

*RL 101 Study Group — Colby Ziyu Wang @ SparkCraft / Hosted by AI Scholars*
*Notes: Alp Guneysel*

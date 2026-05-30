# IV.J. Q-Learning for Foundation Model Alignment {#sec-iv-j}

This section surveys the application of Q-learning to *foundation
model alignment* — the fine-tuning of large language models (LLMs)
and vision-language models (VLMs) toward human preferences,
constrained behaviors, or downstream task performance. Where the
preceding axis-sections (§IV.A-H) treat weaknesses of vanilla
Q-learning that manifest in classical control and Atari, this
section addresses a regime where Q-learning has become a foundational
optimization technique for a model class that the classical
reinforcement learning literature did not originally target. The
section follows the same five-part template as the axis-sections:
weakness, solution families, trade-offs, empirical evidence, open
questions; a sixth comparison summary closes.

### A. The Weakness

Foundation-model alignment differs from classical Q-learning in
three structural ways that change which classical mechanisms apply
and which require re-derivation.

First, the *action space* is the model's output vocabulary at each
token position — typically $10^4$ to $10^5$ tokens — and trajectories
are sequences of token-level decisions over hundreds to thousands of
steps. Per-token Q-learning therefore operates in a high-dimensional
discrete action space with extreme sparsity: at each step, the
overwhelming majority of actions are vocabulary entries the model
would never plausibly emit, and the *effective* action space is the
small subset of tokens with non-negligible policy probability.
Standard $\max_a Q(s, a)$ requires evaluating Q across the full
vocabulary, which is prohibitive for online interaction but
tractable for offline / off-policy settings where the next-state
Q-values can be precomputed.

Second, the *reward signal* is typically derived from a separately
trained reward model (in reinforcement learning from human feedback,
RLHF) or from preference comparisons (in direct preference
optimization, DPO). The reward is dense in episode-aggregate but
sparse in per-token assignment: an entire generation receives a
single scalar score with no decomposition over the tokens that
produced it. This is a sparse-reward credit-assignment regime
(§IV.D) at extreme scale, with the additional complication that the
reward function itself is a learned model with its own approximation
error and adversarial sensitivity.

Third, the *base policy* — the pretrained LLM — is itself a strong
prior that must be preserved during alignment to avoid catastrophic
loss of fluency, instruction-following capability, or factual
grounding. RLHF and DPO both regularize the aligned policy via a KL
penalty against the reference (pretrained) policy. This *KL-
regularized* setting is structurally distinct from standard
Q-learning: the value function being optimized is not the
unregularized return but a return-minus-KL-penalty objective, and
Q-learning adapted to this setting must be modified to respect the
KL constraint.

These three structural differences explain why Q-learning's role in
foundation-model alignment was initially eclipsed by policy-gradient
methods (PPO in early RLHF pipelines, DPO for likelihood-based
alternatives). Value-based methods have since returned as a
recognized sub-area, both because of their off-policy sample
efficiency on fixed datasets and because the KL-regularized setting
admits clean theoretical treatment.

### B. Solution Families

**B.1. Q-Transformer.** Q-Transformer [@chebotar_2023_qtransformer] is the
canonical large-scale Q-learning architecture for sequence-based
control. The method treats the action space as discrete-tokenized
(continuous control actions are discretized into per-dimension token
bins) and trains an autoregressive transformer Q-network to predict
the next action token conditioned on observation history. The
training objective is conservative Q-learning (§IV.E) adapted to the
sequence-token setting: the Q-function is trained to value
in-distribution action tokens highly and out-of-distribution tokens
poorly. While Q-Transformer was developed for robotic manipulation
rather than for natural-language alignment, its architecture and
training recipe established the technical template that subsequent
language-alignment methods build on: autoregressive Q-prediction
over token sequences, with conservative penalties against unseen
tokens to prevent extrapolation error.

**B.2. ShiQ (Shifted-Q for language model logits).** ShiQ adapts
Q-learning directly to the autoregressive language model setting by
treating the model's per-token output *logits* as Q-value estimates,
with a single bias correction that makes the logit-as-Q
interpretation theoretically defensible. The key derivation begins
from the Bellman equation in the KL-regularized setting:

$$
Q_\text{reg}(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}}\Bigl[\log \sum_a \pi_\text{ref}(a \mid s_{t+1}) e^{Q_\text{reg}(s_{t+1}, a) / \beta}\Bigr] - \beta \log \pi_\text{ref}(a_t \mid s_t),
$$

where $\pi_\text{ref}$ is the reference policy and $\beta$ controls
the KL strength. ShiQ shows that a learned Q-function in this
setting can be reparameterized as a *shifted* version of the
language model's logits, where the shift accounts for the reference-
policy contribution. The shifted-logit Q-values can then be trained
via a standard temporal-difference loss against off-policy data,
enabling sample-efficient token-wise learning without the online
sampling required by policy-gradient methods. The method is
mechanically off-policy and structurally compatible with the
KL-regularized RLHF objective, removing the principal computational
cost of policy-gradient RLHF (online rollouts during training).

**B.3. VLM Q-Learning.** VLM Q-Learning extends the off-policy
language-model alignment framework to vision-language models, where
the input modality includes images alongside text and the action
space spans multimodal token vocabularies. The method trains a
multimodal transformer Q-function on a fixed dataset of (image,
text, action, outcome) tuples, with the Q-targets derived from a
reward model trained on preference data. The empirical contribution
is that vision-language models can be aligned to interactive
decision-making tasks (instruction-following in visual environments,
multi-turn dialogue with image grounding) directly from offline
data, including from *low-quality* trajectories — failed or
partially-failed attempts that policy-gradient methods would
typically discard. The off-policy advantage of Q-learning becomes
particularly valuable here, since high-quality interactive-VLM
trajectories are expensive to collect but unsuccessful trajectories
are abundant.

**B.4. Q$^\sharp$ (Q-sharp) for KL-regularized RL.** Q$^\sharp$ provides a more
theoretically principled treatment of KL-regularized RL by training
the Q-function via *distributional* Q-learning (§IV.D) on aggregated
multi-source data, then guiding the reference policy via the
distributional Q-function rather than its expectation. The method
combines two ideas: the regularized Bellman operator of §B.2 above,
and the Wasserstein-contraction guarantees of distributional Q-
learning (§IV.D, §IV.I.C.1). The result is a KL-regularized RL
algorithm whose policy improvements are guided by the full reward
distribution rather than its mean, which Q$^\sharp$ argues is the appropriate
information signal when the reward model has tail uncertainty.
Empirically, Q$^\sharp$ produces lower-KL fine-tuning under matched reward
improvement, suggesting that distributional information helps the
KL-regularized objective more than the standard return objective.

### C. Trade-offs

- **Off-policy efficiency vs. online sample quality.** Q-learning's
  ability to learn from fixed datasets — including failed
  trajectories — is the principal practical advantage over PPO-based
  RLHF. The trade-off is that the off-policy estimates suffer when
  the deployment distribution drifts from the training-data
  distribution, requiring periodic re-collection. PPO-style methods
  avoid this drift but pay the cost of online rollouts during
  training.

- **Vocabulary-scale Q evaluation.** The $\max_a Q(s, a)$ operation
  over a $10^5$-token vocabulary is expensive. Q-Transformer
  addresses this by discretizing into smaller per-dimension
  vocabularies; ShiQ avoids it by reparameterizing Q as the model's
  logits directly (so the $\max$ is built into the softmax already);
  VLM Q-Learning uses targeted top-K sampling. Each choice trades
  expressiveness for tractability.

- **KL-regularization coupling.** Methods that interpret model
  logits as Q-values (ShiQ) inherit a coupling between the policy
  parameterization and the value function. This simplifies the
  algorithm but creates risk that the same network parameters serve
  both objectives, complicating analysis of which trade-offs the
  network is being asked to make.

- **Reward-model sensitivity.** All methods in this section rely on
  a learned reward model. Q-learning's value-recursion amplifies
  reward-model errors more than per-token policy-gradient methods,
  particularly for tail behaviors where the reward model has high
  uncertainty. This is a known and increasingly studied failure mode
  of value-based alignment.

### D. Empirical Evidence

The empirical regime here is distinct from the Atari and tabular
benchmarks of §V and §VI. Reported numbers come from preference-
based reward improvements on instruction-following benchmarks
(AlpacaEval, MT-Bench, IFEval), from task success rates on
interactive-VLM benchmarks (VisualWebArena, OSWorld), and from
KL-divergence-vs-reward Pareto curves that characterize the
fine-tuning trajectory.

The relative empirical positioning is roughly as follows:

- **Q-Transformer** dominates large-scale offline robotic learning
  benchmarks (RT-2 follow-ups, multi-task manipulation), confirming
  the architecture's scalability to large action and observation
  spaces.
- **ShiQ** matches or exceeds PPO-RLHF on standard alignment
  benchmarks while requiring substantially less training compute
  per epoch, with the gap largest on benchmarks where high-quality
  human-preference data is limited.
- **VLM Q-Learning** establishes off-policy learning from low-
  quality VLM trajectories as a working alignment recipe; absolute
  performance on interactive benchmarks remains below online-method
  ceilings, but the cost-effectiveness for marginal data quality is
  substantially better.
- **Q$^\sharp$** improves Pareto efficiency in the KL-vs-reward trade-off,
  with the largest gains in settings where the reward model has
  measured tail uncertainty (multi-step reasoning tasks, code
  generation).

A consistent caveat across the empirical literature is that
foundation-model alignment benchmarks are themselves under-developed:
the reward signal is a learned model, the held-out evaluation often
uses the same family of judges as the training reward model, and
seeds of variation in reward-model training induce substantial
downstream variance. The methodological-rigor concerns of §V.G and
§VI.E apply here with double force.

### E. Open Questions

1. **Online policy correction at deployment time.** Foundation
   models are deployed at scale to user populations whose preferences
   differ from any single fine-tuning dataset. A theory of Q-learning-
   based *online* preference correction — adapting the deployed
   model's behavior to observed user feedback in production — is
   nascent. The classical online learning literature applies in
   principle but encounters the regularization and stability concerns
   of §IV.H at unprecedented scale.

2. **Reward-model robustness as an axis problem.** The reward
   model's failure modes (adversarial sensitivity, distribution
   shift, value-misspecification at the tails) compound through
   Q-learning's value recursion. Whether reward-model robustness
   should be treated as a *separate* axis or as a sub-problem of
   §IV.E distribution shift is open.

3. **Distributional alignment beyond Q$^\sharp$.** Q$^\sharp$'s use of distributional
   information for KL-regularization is an early example; broader
   integration of distributional Q-learning with preference-based
   reward signals — including multi-modal preference distributions
   that classical reward modeling cannot represent — is a productive
   research direction.

4. **Value-based methods for in-context alignment.** The in-context
   Q-learning thread of §IV.G.B.3 (SICQL, ICQL) produces transformer
   Q-functions that adapt at inference time without parameter
   updates. Whether this regime extends to foundation-model alignment
   — adapting the fine-tuned alignment behavior to a new context at
   inference time, without re-training — is one of the most
   commercially relevant open questions in this section.

5. **Theoretical guarantees in the KL-regularized regime.** The
   pessimism-based offline-RL theory of §IV.I.C.3 (Jin et al. 2021)
   does not yet apply cleanly to the KL-regularized objective. A
   pessimism-style bound for KL-regularized Q-learning would put
   foundation-model alignment on the same theoretical footing as
   classical offline RL.

### F. Comparison Summary

| Method (year) | Mechanism | Off-policy? | KL-regularized? | Primary domain |
|---|---|:---:|:---:|---|
| Q-Transformer (2023) | Discretized-action transformer Q + conservative penalty | $\checkmark$ | partial | Robotic manipulation; multi-task control |
| ShiQ | Logits-as-Q with bias correction; TD loss in KL-reg setting | $\checkmark$ | $\checkmark$ | LLM alignment from offline preference data |
| VLM Q-Learning | Multimodal transformer Q on (image, text, action) tuples | $\checkmark$ | partial | Vision-language interactive tasks |
| Q$^\sharp$ | Distributional Q under KL-regularization on aggregated data | $\checkmark$ | $\checkmark$ | Lower-KL fine-tuning, multi-source preference data |

The four methods are complementary rather than competing: each
addresses a different practical constraint of the foundation-model
alignment regime. The combination of off-policy efficiency and
KL-regularization compatibility is the recurring technical theme;
the architectural choices (autoregressive transformer, multimodal
encoder, distributional output) reflect the specific domain each
method targets.

A reader skimming §IV: this section breaks the eight-axis weakness
framework deliberately. The weaknesses of vanilla Q-learning
identified in §II.B do not map cleanly to foundation-model
alignment, which is a *new* application domain rather than a new
mechanism category. The section is included because it represents
the most significant practical application of Q-learning in the
current AI ecosystem and would be conspicuously absent in a survey
dated 2026. Future surveys may well introduce a ninth weakness
(reward-model robustness, KL-regularized stability) that absorbs
this section into the eight-axis framework, but at the time of
writing the cleanest treatment is as a standalone application
section.

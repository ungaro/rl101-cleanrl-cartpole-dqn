# Section IV.G — Scaling and Slow Adaptation

Addresses **W7** of the eight weaknesses introduced in §II.B. This
section covers two related directions: distributed and parallel
architectures that increase sample throughput by orders of magnitude,
and meta-learning approaches that produce policies adapting quickly
to new tasks. The two directions are unified by the shared diagnosis
that *per-task, single-learner Q-learning is fundamentally
sample-bottlenecked*.

**Status:** first-pass draft pending review by a co-author with
distributed-systems or meta-RL background. Coverage emphasizes
canonical methods; the rapidly-evolving distributed-RL landscape
(IMPALA, SEED RL, Sample Factory, Podracer) is touched only at
cross-references.

---

## IV.G. Scaling and Slow Adaptation

### A. The Weakness

Vanilla Q-learning trains a single agent on a single task using
sequential environment interactions. Two distinct failure modes
follow.

**Scaling.** A single sequential learner is bottlenecked on the rate
at which the environment produces transitions. Even with experience
replay, the agent cannot consume transitions faster than they are
generated. Atari training runs at ~30 FPS in the standard
environment; collecting the 200M frames used in Rainbow [28]
evaluation requires approximately 80 days of wall-clock time on a
single machine. The bottleneck is not compute — modern GPUs train
Q-networks orders of magnitude faster than the environment generates
data — but sample throughput.

**Slow adaptation.** Q-learning trained on task $\mathcal{T}_1$
produces a Q-function specialized to $\mathcal{T}_1$. Adaptation to
related task $\mathcal{T}_2$ requires either retraining from scratch
or initialization from $Q^{\mathcal{T}_1}$ — the latter sometimes
helpful, sometimes actively harmful when the tasks share state but
disagree on reward. Neither approach satisfies the requirement of
online adaptation within a small number of episodes — the regime
where a deployed agent encounters a related-but-novel task.

Methods responding to these weaknesses fall into three families:
*distributed architectures* that parallelize environment interaction
across many actors with a centralized learner; *meta-learning
approaches* that train across a distribution of tasks to produce
fast-adaptation initializations or update rules; and *recurrent
architectures* that handle partial observability and trajectory
context within a single agent.

### B. Solution Families

**B.1. Distributed actor-learner architectures.** Ape-X [Horgan et
al. 2018] decouples experience collection from learning. Many
*actors* (typically 64–256 CPU processes) interact with parallel
environment instances using a (potentially stale) copy of the
$Q$-network, sending transitions to a centralized prioritized replay
buffer. A single *learner* (a GPU process) samples from the buffer,
performs Q-learning updates, and periodically broadcasts updated
parameters back to the actors.

The architecture supports prioritized experience replay (§IV.B) at
distributed scale, with priorities computed by the actors at
generation time. Ape-X achieves $\sim 50\times$ faster wall-clock
training than DQN while substantially improving final performance,
reaching state-of-the-art on the majority of Atari games at the
time of publication.

R2D2 (Recurrent Replay Distributed DQN) [Kapturowski et al. 2019]
extends Ape-X with LSTM-based recurrent agents and a *burn-in*
mechanism for replay: when sampling a trajectory segment from
replay, the first $\ell$ steps are used only to initialize the
recurrent state, and gradient updates apply only to the subsequent
steps. The mechanism resolves the *state staleness* problem in
recurrent replay — stored recurrent states are stale by the time
they are sampled, but a short burn-in re-syncs them.

R2D2 demonstrates substantial gains on Atari games with memory
demands (Solaris, Skiing) where non-recurrent agents plateau. It
also doubles as the scaling case for §IV.C: the larger sample
throughput enables exploration strategies (R2D2 uses a fixed-$\epsilon$
mixture across actors with $\epsilon$ values logarithmically spaced
in $[0.4, 4 \times 10^{-3}]$) that would be infeasible in
single-agent training.

**B.2. Bandit-controlled exploration meta-policies.** Agent57 [Badia
et al. 2020] is the first method to achieve human-normalized
performance above 100% on *all* 57 Atari games, including the
canonical hard-exploration suite (Montezuma's Revenge, Pitfall!,
Skiing, Solaris). Agent57 builds on R2D2 and adds:
- A family of policies parameterized by exploration intensity
  $\beta$ and discount $\gamma$, ranging from short-horizon
  exploitative to long-horizon exploratory;
- A multi-armed bandit at the actor level that selects which
  policy parameters to roll out, with rewards based on undiscounted
  episode return — exploiting more when exploitation is paying off
  and exploring more when it is not;
- The Never Give Up (NGU) [Badia et al. 2020] intrinsic
  motivation module providing exploration bonuses based on
  episodic memory.

Agent57's mechanism is fundamentally cross-axis: it addresses
exploration (§IV.C) via bandit-controlled policy selection,
sample efficiency (§IV.B) via distributed actor pools, and slow
adaptation (this section) via a portfolio of policies rather than a
single learned policy. The pattern — *cross-axis composition* — is
the dominant pattern in frontier distributed agents and points
toward a hybrid taxonomy that the problem-first organization
explicitly supports.

**B.3. Meta-learning on the Q-function.** MAML [Finn et al. 2017]
applied to Q-learning [Mendonca et al. 2019, related work] trains
across a task distribution $p(\mathcal{T})$ to find an
initialization $\theta_0$ from which a few gradient steps on any
sampled task produce a good task-specific Q-function. The
meta-objective is

$$
\theta_0^\ast = \arg\min_{\theta_0} \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}\bigl[\mathcal{L}_\mathcal{T}(\theta_0 - \alpha \nabla_{\theta_0} \mathcal{L}_\mathcal{T}(\theta_0))\bigr],
$$

requiring second-order gradients through the inner-loop adaptation.
Reptile-Q [Nichol et al. 2018, RL extension] proposes a first-order
approximation that retains most of the empirical performance at
substantially lower compute.

Meta-RL on the Q-function has produced strong results on small-scale
benchmarks (MetaWorld, Meta-MuJoCo) but has not transferred to
Atari-scale evaluation. The compute cost of meta-training on a
distribution of Atari games approaches prohibitive scales; the more
productive path has been the *implicit* meta-learning of Agent57
(a single agent trained to handle the entire Atari distribution at
once) rather than the *explicit* meta-learning of MAML.

**B.4. Recurrent architectures for partial observability.** Deep
Recurrent Q-Network (DRQN) [Hausknecht & Stone 2015] introduces an
LSTM layer into the DQN architecture, replacing the first
fully-connected layer. The recurrent network maintains a hidden
state $h_t$ summarizing the trajectory so far; the Q-function
becomes $Q(h_t, a)$ rather than $Q(o_t, a)$, addressing partial
observability.

DRQN is included in this section rather than alongside DQN
because its primary contribution is *contextual adaptation* — the
ability to condition on trajectory history — which sits closer to
slow-adaptation and partial-observability concerns than to the core
Q-learning mechanism. The connection to R2D2 is direct: R2D2's
recurrent architecture is DRQN's mechanism deployed at distributed
scale with burn-in correction for replay.

**B.5. Synchronous parallelism (cross-reference to §IV.H).** PQN
[55] adopts a different parallelism model: synchronous vectorized
environments where a single learner processes $B$ environments per
step, accumulating gradients across the batch. The trade-off
relative to Ape-X is:
- PQN is simpler (no actor-learner communication), runs on a single
  machine, and removes the staleness inherent in actor-learner
  systems;
- Ape-X scales beyond single-machine limits, supports prioritized
  replay, and admits heterogeneous actor configurations
  (e.g., different exploration parameters per actor as in R2D2/
  Agent57).

PQN's argument that synchronous parallelism + normalization
suffices for Atari-scale training is one of the strongest recent
challenges to the actor-learner orthodoxy. The right architecture
likely depends on whether the cost structure is compute-dominated
(favoring synchronous) or environment-dominated (favoring
asynchronous).

### C. Trade-offs

- **Throughput vs. staleness.** Asynchronous actor-learner systems
  scale throughput linearly in actor count but introduce *parameter
  staleness*: actors collect data using parameters $k$ updates
  behind the learner. The staleness is empirically tolerable up to
  modest $k$ but becomes destabilizing at scale. Synchronous
  parallelism eliminates staleness at the cost of throughput
  scaling.
- **Exploration heterogeneity.** Distributed actor pools admit
  *heterogeneous exploration*: different actors can run different
  exploration strategies in parallel. R2D2's $\epsilon$-mixture and
  Agent57's bandit-controlled policy portfolio both exploit this.
  Single-learner systems cannot replicate the mechanism.
- **Meta-training cost.** MAML-style meta-RL on Q-learning requires
  many full inner-loop adaptations per outer-loop step. The compute
  cost can exceed the equivalent single-task training by 10–100×.
  The relevant question for adoption is whether the few-shot
  transfer gain compensates for the meta-training cost. Empirical
  evidence is mixed.
- **Implicit vs. explicit meta-learning.** Agent57 demonstrates
  that a sufficiently capable single agent trained across a task
  distribution can match or exceed explicit meta-learners on the
  same distribution. The trade-off is *what is meta-learned*
  (an initialization, an update rule, a policy portfolio) and
  *how it is used at deployment* (gradient-step adaptation,
  forward-pass adaptation, bandit selection).

### D. Empirical Evidence

The empirical case for this section spans multiple benchmarks
because the methods target different combinations of axes. The
strongest single empirical result is Agent57's solution of *all
57 Atari games* to above human-normalized performance, including
games where every prior method had scored zero or near-zero
(Montezuma's Revenge to 9,352; Pitfall! to 41,313; Skiing to
-4,202.6 — above human's -3,629).

Selected results from the distributed-RL line on Atari median
human-normalized score after the indicated training scale:

| Method | Frames | Median HNS | Notes |
|---|---|---|---|
| Nature DQN | 200M | 79% | Single-learner baseline |
| Rainbow | 200M | 223% | Single-learner combination |
| Ape-X | 22B | 434% | Distributed; 100+ actors |
| R2D2 | 10B | 1920% | Distributed + recurrent |
| Agent57 | 78B | 4766% | Distributed + bandit-controlled exploration |
| PQN | 200M | 220% | Single-machine synchronous |

Two observations:

First, **the move to distributed training accounts for ~10× the
performance improvement of any single algorithmic innovation in
this paper's scope.** The Ape-X → R2D2 → Agent57 progression is
the most consistent single source of performance gain in the
field since DQN. The empirical lesson is that *sample throughput
is, in practice, the binding constraint on Atari performance* —
the algorithmic axes addressed elsewhere in this paper become
secondary at sufficient scale.

Second, **PQN's single-machine result matching Rainbow** suggests
that the *algorithmic* axes are far from saturated, only that the
single-machine compute budget bounds what can be demonstrated.
PQN at 200M frames matches Rainbow at 200M frames; what PQN at
22B frames would achieve is empirically untested.

Meta-RL Q-learning results are reported on smaller benchmarks
(MetaWorld, RL-Bench, Meta-MuJoCo) where direct comparison to
distributed Atari is not possible. The within-benchmark evidence
shows meta-Q methods adapting in 5–50 episodes to held-out tasks
where single-task baselines require thousands. The transfer to
Atari-scale benchmarks has not occurred.

### E. Open Questions

1. **Compute-vs.-sample Pareto.** The Agent57 result raises the
   question of whether *every* algorithmic innovation in this paper
   would close the human-performance gap on Montezuma if given
   78B training frames. If so, the algorithmic taxonomy becomes
   primarily relevant in the compute-constrained regime; if not,
   the gap diagnoses which methods are truly addressing the
   exploration weakness vs. compensating for it via scale. No
   systematic study of this question exists.

2. **Heterogeneous task distributions.** Meta-RL Q-learning has
   succeeded on benchmarks where the task distribution is narrow
   (parametric variations of a single environment). Performance
   on heterogeneous distributions — different state spaces,
   different action spaces, different reward structures — remains
   weak. Whether the meta-learning framework can be extended to
   substantially heterogeneous task families is the question
   gating the practical adoption of meta-Q methods.

3. **Compute requirement for problem-first ablations.** This
   paper's structural argument — that methods target distinct
   weaknesses with distinct mechanisms — would be strongest if
   supported by ablations holding compute fixed and measuring
   axis-specific performance gains. At Atari-scale, such
   ablations are prohibitively expensive. A reduced-scale
   benchmark designed specifically for axis-stratified evaluation
   would be a substantial methodological contribution to the
   field.

4. **The actor-learner-synchronous tradespace.** Ape-X-style
   asynchronous parallelism and PQN-style synchronous vectorization
   represent two points on a tradespace whose interior is largely
   unexplored. Hybrid architectures — synchronous within an
   actor pool, asynchronous between pools — exist (IMPALA, Sample
   Factory) but the formal characterization of when each
   architecture is preferred remains incomplete.

---

*Notes for integration:*
- This is the third and largest modern-RL addition. The
  bibliography needs ~6 new entries: Ape-X (arXiv:1803.00933), R2D2
  (DeepMind blog + ICLR 2019), Agent57 (arXiv:2003.13350), NGU
  (arXiv:2002.06038), MAML (arXiv:1703.03400), DRQN
  (arXiv:1507.06527, already cited).
- DRQN is relocated from §IV.B (Q-Function Computation
  Innovations) to §IV.G. The re-interpretation as a
  contextual-adaptation method rather than an architectural
  variant is the load-bearing claim of subsection B.4.
- The cross-axis nature of Agent57 is the strongest empirical
  argument for the problem-first organization. Agent57 deliberately
  composes mechanisms from §IV.B (sample efficiency via
  distributed replay), §IV.C (exploration via bandit-controlled
  policy portfolio + NGU intrinsic motivation), and this section
  (slow adaptation via implicit task-distribution training).
  Frontier agents are increasingly cross-axis; this section makes
  the pattern visible.
- IMPALA, SEED RL, Sample Factory, Podracer are mentioned only at
  cross-reference; full coverage of the distributed-RL architecture
  literature would substantially expand the section. Defer to
  domain owner judgment on whether to include.
- Cross-references: §IV.B (PER at distributed scale), §IV.C
  (exploration heterogeneity), §IV.H (PQN's synchronous
  parallelism).

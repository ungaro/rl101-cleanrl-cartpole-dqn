# VIII. Conclusion and Future Work {#sec-viii}

### A. Summary

This paper presents a problem-first survey of Q-learning and deep
Q-learning, reorganizing three decades of methodological development
around the eight foundational weaknesses of vanilla Q-learning
introduced in §II.B: overestimation bias, sample inefficiency,
brittle exploration, reward sparsity and credit assignment,
distribution shift, multi-agent coordination, slow adaptation, and
function-approximation instability.

For each weakness, §IV surveys the families of methods that respond:

- **§IV.A** — overestimation control via decoupling (Double Q,
  Double DQN) and ensembles (EBQL, REDQ).
- **§IV.B** — sample efficiency via prioritized sampling (PER),
  demonstration augmentation (DQfD), memory-efficient consolidation
  (MeDQN), and goal relabeling (HER).
- **§IV.C** — directed exploration via noise injection (NoisyNet,
  Parameter Space Noise), ensemble disagreement (Bootstrapped DQN,
  UCB Q-Ensemble), belief modulation (CBDQ, PSDQN), and intrinsic
  motivation (RND, Go-Explore).
- **§IV.D** — credit assignment via multi-step returns and the
  distributional family (C51, QR-DQN, IQN, FQF).
- **§IV.E** — offline RL via policy constraint (BCQ, AWAC, BRAC),
  value penalty (CQL), expectile regression (IQL), and ensemble
  diversification (EDAC).
- **§IV.F** — cooperative value decomposition via additive (VDN),
  monotonic mixing (QMIX), duplex dueling (QPLEX), and constraint-
  free (QTRAN) factorizations.
- **§IV.G** — scaling and adaptation via distributed actor-learner
  architectures (Ape-X, R2D2), bandit-controlled exploration
  meta-policies (Agent57), and meta-learning on the Q-function.
- **§IV.H** — function-approximation stability via target networks,
  architectural decomposition (Dueling), normalization recipes (PQN),
  and consolidation losses (MeDQN, Munchausen DQN).

Sections V and VI present axis-stratified empirical evidence,
showing that methods targeting weakness $W_i$ excel on the task
categories that diagnose $W_i$ and remain near baseline elsewhere.
Section VII analyzes algorithmic coverage across six open-source
repositories, identifying nine methods absent from all surveyed
codebases — a gap that motivates the principal forward-looking
deliverable of this paper.

### B. A community Q-learning repository

The repository coverage analysis of §VII surfaces a concrete
implementation gap: nine methods covered in §IV — DRQN, CBDQ, DQfD,
MeDQN, Bootstrapped DQN, UCB Q-Ensemble, EBQL, Posterior Sampling
DQN, and PQN — are absent from *every* surveyed open-source
repository. Across the new axes introduced in §IV.E, §IV.F, and
§IV.G, the gap is broader still: most offline-RL, multi-agent
value-decomposition, and distributed-Q methods have no
implementation in any of the six repositories analyzed.

We propose the development of a **dedicated, community-maintained
repository focused exclusively on Q-learning and its deep variants**.
The repository would have four design priorities:

1. **Axis-aware coverage** — implementations organized by the eight
   weakness axes, prioritizing methods absent from existing
   repositories, with a coverage matrix maintained as the public
   roadmap.
2. **Unified training scripts** — a common interface across methods
   permitting like-for-like comparison under matched evaluation
   protocols.
3. **Standardized benchmarks** — Atari, classic control, D4RL, and
   SMAC included as canonical benchmark suites, with reference
   results for each implemented method.
4. **Empirical reproducibility audits** — each method release gated
   on reproduction of the original paper's headline results, in the
   style of Hundal et al. [@hundal_2025_interchangeable].

The repository's initial roadmap would prioritize the nine methods
in §VII.C as the first implementation targets, followed by the
offline-RL family (CQL, IQL, EDAC), the multi-agent value-
decomposition family (QMIX, QPLEX, VDN), and the
distributed-Q family (Ape-X, R2D2, Agent57). Cross-references
between the repository's documentation and this paper's §IV
sections would let practitioners navigate from "I face problem $X$"
(this paper's axis) to "available implementations addressing $X$"
(the repository's coverage matrix).

### C. Open research directions, by axis

The "Open Questions" subsections (E) of each §IV axis-section
collectively summarize the field's frontier. We highlight four
cross-axis directions of particular promise:

**The compute-vs.-algorithmic tradeoff.** The empirical pattern in
§V suggests that distributed scaling (§IV.G) accounts for the
largest single performance increase in deep RL since DQN, but PQN's
single-machine results suggest the algorithmic axes are far from
saturated. A systematic study of which weaknesses are
*compute-resoluble* versus *algorithmically-resoluble* — at fixed
compute budgets — would clarify where research effort is best
directed.

**Cross-axis composition.** Frontier agents (Agent57, NGU)
explicitly compose mechanisms from multiple axes. A formal account
of which axis combinations are synergistic, redundant, or
antagonistic would convert the empirical pattern into a design
principle. The Rainbow ablation [@hessel_2018_rainbow] provides one such account on
a smaller method set; a broader cross-axis composition study would
extend it.

**The offline-online interface.** Methods bridging offline and
online learning (AWAC, Cal-QL) address a regime increasingly central
to practical deployment but theoretically underexplored. The
interface is itself a candidate weakness axis the field may
recognize in coming years.

**Q-learning under real-world distribution shift.** §IV.E treats
distribution shift as a *training* regime: the data was collected by
one or more behavior policies and is then fixed. A deployment-side
analog — the learned policy meets a state distribution different
from the one it was trained on — is largely unaddressed. Sim-to-real
transfer in robotic Q-learning, OOD robustness against perturbations,
and online policy correction at deployment time all live in this
regime. The newer benchmarks of §V.I (ProcGen for procedural
variation, CARL for context generalization, RLBench for
sim-to-real) begin to provide evaluation infrastructure, but no
unified Q-learning treatment of *deployment* distribution shift
exists. This is the axis where the gap between Atari-evaluated
methods and real-world deployment is most visible, and where
Q-learning's value-based perspective could productively engage with
robotics, control, and operations-research literature that has
developed parallel tools.

### D. Limitations and Ethics

This is a survey rather than a deployed system, so the relevant
considerations are scholarly and community-facing rather than
direct model-harm concerns. We surface five limitations explicitly.

**Benchmark monoculture.** Tables II–III's reliance on Atari and
§VI's tabular Gymnasium environments reflects the field's
historical convention. As documented in §V.H, Atari is a strong
benchmark for a specific set of weaknesses (exploration, credit
assignment, function-approximation stability) and a weak one for
others (sample efficiency at small budgets, generalization across
procedural variation, continuous control, real-world deployment).
A survey that elevates Atari evidence to load-bearing status
inherits these gaps. We have partly mitigated this through §V.I's
coverage of ProcGen, NetHack, BSuite, D4RL, and SMAC and through
§IV.E and §IV.F's adoption of D4RL and SMAC respectively, but the
breadth of evaluation evidence remains uneven across the eight
axes.

**Compute inequality.** Several of the methods covered (Agent57's
78B training frames, R2D2's 10B, large in-context Q-learning
agents) require compute budgets accessible to only a handful of
industrial labs. By including these methods alongside academically
reproducible alternatives without flagging the gap, a survey
risks implicitly endorsing a research culture where frontier work
is structurally inaccessible to most readers. §IV.G's discussion of
the compute-versus-algorithmic trade-off and §V.H.5 on compute
scale begin to address this, but the field-wide tension between
*algorithmic discovery* and *compute-driven discovery* is
underdeveloped here and warrants explicit treatment in future
surveys.

**Citation bias in narrative reviews.** Any narrative survey that
reduces a large field to "canonical methods" can inadvertently
erase negative results, replication studies, and less-publicized
alternatives — even when, as here, the source selection criteria
are documented (§III.A). We cite the prior-art sweep in
`07-prior-art-sweep.md` as the audit trail for our differentiation
claim, and Appendix A's legacy indexer to allow navigation through
the conventional taxonomy, but we cannot eliminate the bias
inherent in narrative selection. Readers seeking a more systematic
review-of-reviews should consult the Springer NCAA 2026 offline-RL
distribution-shift survey and the broader-RL surveys of Ghasemi
2024/2025 and Murphy 2024/2025 as complementary navigations of
overlapping terrain.

**Stale tooling comparisons.** The repository comparison of §VII
is pinned to repository state as of early 2026. Implementation
landscapes shift quickly; readers consulting this paper in 2027 or
later should treat Table VII as a snapshot rather than current
ground truth. The proposed Q-learning–specific repository (§VIII.B)
is intended in part as a mechanism for maintained tooling
comparison rather than as a one-shot artifact.

**Frontier methods overstated relative to settled techniques.**
This paper devotes substantial space to in-context Q-learning
(§IV.G.B.3), flow-matching policies (§IV.E.B.5), and recent
theoretical advances (§IV.I.C) — all subject to ongoing community
consensus formation. Some claims in those subsections may be
revised as the field consolidates. We have flagged the relevant
sections with explicit "active research area" language where
applicable, but readers should treat any specific claim about
post-2024 work as more uncertain than the well-established results
in §IV.A–D and §IV.H.

A small note on broader ethical considerations: Q-learning's
expanding role in robotics, recommender systems, and language-model
alignment carries downstream policy implications that this survey
does not address. A reader interested in the responsible deployment
of value-based RL methods should consult the broader RL-safety and
RL-policy literature (Saunders et al. 2023, Anderljung et al.
2024) as complementary material.

### E. Closing

Q-learning's longevity as a foundational paradigm rests on its
combination of conceptual simplicity and methodological extensibility.
The thirty-five years since [Watkins 1989] have produced a tooling
ecosystem rich enough to deserve organized treatment but specialized
enough that no general-RL survey can capture its texture. This paper
offers the first multi-axis problem-first synthesis of that
ecosystem, paired with empirical and implementation analyses that
support the structural claim.

The methodological landscape this paper traces is not closed. New
axes will emerge — the offline-online interface and the meta-RL
boundary are two candidates — and existing axes will continue to
admit new mechanisms. The problem-first organization is intended
to be extensible: new methods slot into existing axes, and new
axes can be added as the field recognizes them. The framework's
value is precisely that it survives those additions.

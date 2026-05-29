# Method → Axis Remap (working notes)

Every method in the current draft (plus the modern-RL methods we
propose adding) mapped to its new axis-section in the problem-first
structure of `03-new-outline.md`.

**Status:** this file was the working-notes scratchpad for the
remap. The canonical version of the mapping is now
[`draft/A-legacy-indexer.md`](draft/A-legacy-indexer.md), which adds
a reverse view (legacy category → axis distribution) and treats the
new methods consistently. Prefer Appendix A for handoff to the team;
this file is preserved as the design notes that informed it.

The "Old §" column points to the current draft section. The "New §"
column points to the proposed new section. The "Secondary axes" column
notes when a method appears (briefly) in additional sections — these
cross-references are exactly the "synthesis the reviewers want."

Legend for new sections:
- **IV.A** Overestimation Bias
- **IV.B** Sample Inefficiency
- **IV.C** Brittle Exploration
- **IV.D** Reward Sparsity & Credit Assignment
- **IV.E** Distribution Shift (Offline RL) — *new*
- **IV.F** Multi-Agent Coordination — *new*
- **IV.G** Scaling & Slow Adaptation — *new + DRQN*
- **IV.H** Function-Approximation Instability
- **V** Foundations (pre-deep-RL methods)

---

## Methods already in the draft

| Method | Year | Old § | New § (primary) | Secondary axes | Notes |
|---|---|---|---|---|---|
| Parameter Space Noise | 2017 | IV.A (Statistical) | **IV.C** | — | Exploration mechanism, not value uncertainty |
| NoisyNet | 2018 | IV.A | **IV.C** | — | Same as above |
| C51 (Distributional) | 2017 | IV.A | **IV.D** | IV.A (uncertainty) | Distribution as credit-assignment signal is the *load-bearing* reading |
| QR-DQN | 2018 | IV.A | **IV.D** | IV.A | |
| IQN | 2018 | IV.A | **IV.D** | IV.A | |
| FQF | 2019 | IV.A | **IV.D** | IV.A | |
| Double Q-Learning | 2010 | IV.B (Q-Function) | **IV.A** | V (foundational) | The canonical overestimation fix |
| Nature DQN | 2015 | IV.B | **IV.H** | V | Target network is the *stability* contribution, not the value-comp one |
| DRQN | 2015 | IV.B | **IV.G** | — | Partial observability → adaptation/architecture axis |
| Double DQN | 2016 | IV.B | **IV.A** | — | |
| Dueling DQN | 2016 | IV.B | **IV.H** | IV.A (incidentally) | Re-interpret as a stability/decomposition contribution |
| Rainbow | 2018 | IV.B | **IV.B** | IV.A, IV.C, IV.D, IV.H | Lives in *Sample Inefficiency* because PER and multi-step are the largest ablation contributors; cross-references everywhere |
| MCTS for FrozenLake | 2024 | IV.B | V (kept as planning baseline) | — | Doesn't really fit any axis; honest about that |
| Cognitive Belief-Driven Q (CBDQ) | 2025 | IV.B | **IV.C** | IV.A | Belief over actions = directed exploration with bias-reduction side effect |
| DQN (2013) | 2013 | IV.C (Memory/Replay) | **IV.B** | V | Replay buffer = sample efficiency baseline |
| Prioritized ER (PER) | 2016 | IV.C | **IV.B** | — | |
| DQfD | 2018 | IV.C | **IV.B** | — | Demonstrations as sample efficiency boost (and exploration in sparse settings) |
| MeDQN | 2023 | IV.C | **IV.B** | IV.H (consolidation loss = stability) | |
| Bootstrapped DQN | 2016 | IV.D (Ensemble) | **IV.C** | IV.A | Ensembles for deep exploration is the *load-bearing* reading |
| Ensemble Bootstrapped Q (EBQL) | 2021 | IV.D | **IV.A** | IV.C | Ensembles for bias reduction is *the* contribution |
| UCB Q-Ensemble | 2018 | IV.D | **IV.C** | IV.A | UCB framing makes it exploration-first |
| Value/Policy Iteration (VI/PI/MPI/CVPI) | 1960s | IV.E (Model-Based) | **V** | — | Foundational planning baselines; not an axis fix |
| Bayesian Q-Learning | 1998 | IV.E | **V** | IV.C (posterior sampling is directed exploration) | Cross-references to IV.C |
| Expected SARSA | 2009 | IV.E | **V** | — | Foundational variance-reduction; not really model-based |
| Posterior Sampling DQN | 2023 | IV.E | **IV.C** | IV.A | Posterior sampling is directed exploration |
| Q-Learning (Watkins) | 1992 | IV.F (Minimal) | **V** | — | The thing every other section is fixing |
| SARSA | 1994 | IV.F | **V** | — | |
| Multi-Step Q-Learning (Q(λ)) | 1996 | IV.F | **IV.D** | V | n-step is the original credit-assignment fix |
| NFQ | 2005 | IV.F | **V** | IV.B | Batch updates as sample-efficiency precursor |
| PQN | 2025 | IV.F | **IV.H** | IV.G | "Simplifying deep Q without target networks" is fundamentally a stability + scale story |

**Coverage check:** every method in the draft has a home. No method
disappears.

---

## Methods we propose adding (modern RL)

These are the methods that currently have *no slot* in the draft's
taxonomy. Adding them to the existing six categories would be
forced-fit; the new structure has named axis-sections where they
belong.

| Method | Year | New § | Why it matters |
|---|---|---|---|
| Maximin Q-Learning | 2020 | **IV.A** | Generalization of double-Q to k-estimator min; rounds out the overestimation family |
| REDQ | 2021 | **IV.A** | Random ensemble distillation — under-estimation pushback; the bias-variance trade-off made explicit |
| HER (Hindsight Experience Replay) | 2017 | **IV.B** | Sparse-reward sample efficiency via goal relabeling |
| RND (Random Network Distillation) | 2018 | **IV.C** | Curiosity-driven exploration; current go-to for Montezuma-class games |
| Go-Explore | 2019/2021 | **IV.C** | Archival exploration; settles the Montezuma debate |
| CQL (Conservative Q-Learning) | 2020 | **IV.E** | Canonical offline-RL Q-method |
| IQL (Implicit Q-Learning) | 2021 | **IV.E** | Avoids OOD actions via expectile regression |
| BCQ (Batch-Constrained Q) | 2019 | **IV.E** | The pre-CQL approach; explicit OOD handling |
| EDAC | 2021 | **IV.E** | Ensemble + diversification penalty; bridges offline RL to ensemble Q |
| AWAC | 2020 | **IV.E** | Advantage-weighted offline-to-online |
| VDN (Value Decomposition Networks) | 2018 | **IV.F** | Additive Q decomposition for cooperative MARL |
| QMIX | 2018 | **IV.F** | Monotonic mixing — the dominant value-decomp method |
| QPLEX | 2020 | **IV.F** | Duplex dueling extension of QMIX |
| QTRAN | 2019 | **IV.F** | Removes monotonicity constraint |
| Ape-X | 2018 | **IV.G** | Distributed prioritized replay — the scale story |
| R2D2 | 2019 | **IV.G** | Recurrent distributed replay |
| Agent57 | 2020 | **IV.G** | Bandit-controlled exploration meta-policy; first to beat human on all 57 Atari games |
| MetaQ / MAML on Q | 2017–2020 | **IV.G** | Meta-RL on value learning |
| Munchausen DQN | 2020 | **IV.H** | Add log-policy bonus — surprisingly strong, gets at stability via implicit KL regularization |
| Layer Norm + Residual variants (Smith et al. 2022/23) | 2022–23 | **IV.H** | Recent stability recipes that PQN partially adopts |

**This is ~18 new methods.** Not all need full per-paper treatment —
some get a one-paragraph treatment inside a family ("CQL, IQL, BCQ —
three approaches to the OOD-action problem; for full treatments see
[refs]"). The big lift is fair treatment of **3 canonical methods per
new section** (IV.E: CQL/IQL/BCQ; IV.F: VDN/QMIX/QPLEX; IV.G:
Ape-X/R2D2/Agent57), totaling ~9 methods at full depth.

---

## Cross-section appearances (the synthesis)

Methods that appear in multiple axis-sections, with cross-references,
are where the reviewers' "need conceptual insight" complaint gets
answered most directly. The richest examples:

- **Rainbow** appears in IV.B (primary — PER and multi-step drive the
  ablation), and is cross-referenced from IV.A (Double Q component),
  IV.C (NoisyNet component), IV.D (distributional component), and IV.H
  (target network and dueling components). Rainbow is the *paradigm
  case* of methodological synthesis — the new structure surfaces this
  rather than burying it in a single bucket.
- **PQN** appears in IV.H (primary — its stability recipe is the
  contribution) and IV.G (its parallelism / vectorization). Its
  fundamental claim — "remove target networks and replay" — is a
  *stability* claim, not a category of its own.
- **Bootstrapped DQN / EBQL / UCB Q-Ensemble** all sit in IV.C as
  *ensembles for exploration*, with EBQL cross-referenced from IV.A
  as *ensembles for bias reduction*. The three are not just "Ensemble
  methods" — they target different axes with the same mechanism, and
  the new structure shows that.
- **Distributional methods (C51 / QR-DQN / IQN / FQF)** sit in IV.D
  (richer credit-assignment signal) with explicit cross-reference to
  IV.A (the uncertainty/over-estimation interpretation). We pick IV.D
  as load-bearing because the dominant empirical claim — that
  distributional methods *outperform* Rainbow without its other
  components — is a *signal* claim, not an *uncertainty* claim.

These cross-references are themselves a contribution. A reviewer
reading IV.A → "see also Rainbow in IV.B, EBQL in IV.A primary,
distributional uncertainty interpretation in IV.D" is being shown the
field's connective tissue.

---

## What this doesn't tell us

This remap is a *plan*. It does not validate that:

- The new IV.E/F/G sections can be written to journal-quality depth in
  the time available (domain owners gating)
- Every cross-reference resolves cleanly without prose contortions
  (the worked example in `draft/4a-overestimation-bias.md` is the
  smoke test for this)
- The bibliography-style claim "this is the *physician's* taxonomy" is
  defensible — confirmed by the 2024–2026 prior-art sweep in
  `07-prior-art-sweep.md`; no competing problem-first organization
  found

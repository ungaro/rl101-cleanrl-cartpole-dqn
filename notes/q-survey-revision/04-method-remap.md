# Method → Axis Remap (working notes)

> **Synced to v0.24 (2026-05-30).** Aligned with the ground-truth
> state brief: method-type taxonomy locked, eight-axis (W1–W8)
> structure final, full method index relocated to supplement S2.

Every method in the draft mapped to its axis-section in the
problem-first structure now realized as §IV "Q-Learning Methods by
Weakness" (`draft-tai/`).

**Status:** working-notes scratchpad, kept in sync with the final
draft. The **method-type taxonomy** is now *defined once* at the top
of §IV via a single table: **six method-type categories → which of
the eight axes (W1–W8) they touch**. The "legacy" framing is dead —
there is no "legacy category"; everything is one method-type
taxonomy. The full **~50-method method-type index** (3 tables) now
lives in **supplement S2** (was Appendix A); a compact, category-level
method-type table is in **§IV** itself. This file is the design notes
behind the S2 index and the §IV table, kept current for handoff.

The "Old §" column points to the pre-remap draft section. The
"Axis" column points to the final W1–W8 axis subsection (§IV.A–H).
The "Secondary axes" column notes when a method appears (briefly) in
additional sections — these cross-references are the cross-axis
synthesis surfaced in the §IV cross-axis interaction table.

The eight weakness axes and their §IV subsections (W7 is a COMPOSITE
axis = W7a sample throughput + W7b slow adaptation):
- **W1 → §IV.A** Overestimation Bias
- **W2 → §IV.B** Sample Inefficiency
- **W3 → §IV.C** Brittle Exploration
- **W4 → §IV.D** Reward Sparsity & Credit Assignment
- **W5 → §IV.E** Distribution Shift (Offline RL)
- **W6 → §IV.F** Multi-Agent Coordination
- **W7 → §IV.G** Scaling & Slow Adaptation (W7a throughput + W7b adaptation)
- **W8 → §IV.H** Function-Approximation Instability
- **§IV.I** theoretical advances; **§IV.J** foundation-model alignment
  (emerging direction)
- **§V/§VI** foundations & tabular baselines (pre-deep-RL methods,
  planning oracle separated)

The six method-type categories (defined once in the §IV table) are the
mechanism families — value-computation/bias-correction, replay & data
reuse, exploration, distributional/credit-assignment, offline /
distribution-shift, and architecture/scaling — each mapping to one or
more of the W1–W8 axes above.

---

## Methods already in the draft

| Method | Year | Old § | Axis (§IV primary) | Secondary axes | Notes |
|---|---|---|---|---|---|
| Parameter Space Noise | 2017 | IV.A (Statistical) | **W3 / §IV.C** | — | Exploration mechanism, not value uncertainty |
| NoisyNet | 2018 | IV.A | **W3 / §IV.C** | — | Same as above |
| C51 (Distributional) | 2017 | IV.A | **W4 / §IV.D** | W1 (uncertainty) | Distribution as credit-assignment signal is the *load-bearing* reading |
| QR-DQN | 2018 | IV.A | **W4 / §IV.D** | W1 | |
| IQN | 2018 | IV.A | **W4 / §IV.D** | W1 | |
| FQF | 2019 | IV.A | **W4 / §IV.D** | W1 | |
| Double Q-Learning | 2010 | IV.B (Q-Function) | **W1 / §IV.A** | §V (foundational) | The canonical overestimation fix |
| Double DQN (DDQL) | 2016 | IV.B | **W1 / §IV.A** | — | Deep-RL overestimation fix; lead method of the §IV.A overestimation family |
| Parameterized DQN (PDQN) | 2018 | — | **W1 / §IV.A** | — | Hybrid discrete-continuous; RETAINED in §IV.A (full continuous-action expansion declined for scope) |
| Nature DQN | 2015 | IV.B | **W8 / §IV.H** | §V | Target network is the *stability* contribution, not the value-comp one |
| DRQN | 2015 | IV.B | **W7 / §IV.G** | — | Partial observability → adaptation/architecture axis |
| Dueling DQN | 2016 | IV.B | **W8 / §IV.H** | W1 (incidentally) | Re-interpret as a stability/decomposition contribution |
| Rainbow | 2018 | IV.B | **W2 / §IV.B** | W1, W3, W4, W8 | Lives in *Sample Inefficiency* because PER and multi-step are the largest ablation contributors; cross-references everywhere |
| MCTS for FrozenLake | 2024 | IV.B | §VI (planning baseline) | — | Doesn't really fit any axis; honest about that |
| Cognitive Belief-Driven Q (CBDQ) | 2025 | IV.B | **W3 / §IV.C** | W1 | Belief over actions = directed exploration with bias-reduction side effect; absent from all six repos (§VII) |
| DQN (2013) | 2013 | IV.C (Memory/Replay) | **W2 / §IV.B** | §V | Replay buffer = sample efficiency baseline |
| Prioritized ER (PER) | 2016 | IV.C | **W2 / §IV.B** | — | |
| DQfD | 2018 | IV.C | **W2 / §IV.B** | — | Demonstrations as sample efficiency boost (and exploration in sparse settings) |
| MeDQN | 2023 | IV.C | **W2 / §IV.B** | W8 (consolidation loss = stability) | |
| Bootstrapped DQN | 2016 | IV.D (Ensemble) | **W3 / §IV.C** | W1 | Ensembles for deep exploration is the *load-bearing* reading |
| Ensemble Bootstrapped Q (EBQL) | 2021 | IV.D | **W1 / §IV.A** | W3 | Ensembles for bias reduction is *the* contribution |
| UCB Q-Ensemble | 2018 | IV.D | **W3 / §IV.C** | W1 | UCB framing makes it exploration-first |
| Value/Policy Iteration (VI/PI/MPI/CVPI) | 1960s | IV.E (Model-Based) | **§VI** | — | Planning ORACLE — separated upper bound (full model access), NOT a model-free competitor |
| Bayesian Q-Learning | 1998 | IV.E | **§V** | W3 (posterior sampling is directed exploration) | Cross-references to §IV.C |
| Expected SARSA | 2009 | IV.E | **§VI** | — | Foundational variance-reduction baseline; in the 100-seed tabular experiment |
| Posterior Sampling DQN (PSDQN) | 2023 | IV.E | **W3 / §IV.C** | W1 | Posterior sampling is directed exploration; absent from all six repos (§VII) |
| Q-Learning (Watkins) | 1992 | IV.F (Minimal) | **§V** | — | The thing every other section is fixing |
| SARSA | 1994 | IV.F | **§V/§VI** | — | In the 100-seed tabular experiment (CliffWalking safe path) |
| Multi-Step Q-Learning (Q(λ)) | 1996 | IV.F | **W4 / §IV.D** | §V | n-step is the original credit-assignment fix |
| NFQ | 2005 | IV.F | **§V** | W2 | Batch updates as sample-efficiency precursor |
| PQN | 2025 | IV.F | **W8 / §IV.H** | W7 | "Simplify deep Q without target networks/replay" is a stability claim first; parallelism is the W7 side. LayerNorm-Lipschitz contraction math → supp S5. Absent from all six repos (§VII) |
| QFIX / Q+FIX | 2025 | — | **W6 / §IV.F** | — | Value-factorization fix for cooperative MARL; Q+FIX formula + Dec-POMDP V(h,s) → supp S5 |
| SICQL / ICQL | 2025 | — | **W7 / §IV.G** | §IV.J (FM alignment) | Scaling/adaptation axis; also referenced in §IV.J foundation-model alignment. SICQL/ICQL losses → supp S5 |
| Cal-QL | 2023 | — | **W5 / §IV.E** | — | Calibrated offline-to-online Q; offline/distribution-shift family |
| FQL (Flow Q-Learning) | 2025 | — | **W5 / §IV.E** | — | Flow-based offline Q; offline/distribution-shift family |

**Coverage check:** every method in the draft has a home. No method
disappears. The recent additions (DDQL & PDQN → W1/§IV.A; QFIX/Q+FIX →
W6/§IV.F; PQN → W8/§IV.H; SICQL/ICQL → W7/§IV.G, also §IV.J; CBDQ &
PSDQN → W3/§IV.C; Cal-QL/FQL → W5/§IV.E) are all placed above. Nine
methods (DRQN, CBDQ, DQfD, MeDQN, Bootstrapped DQN, UCB Q-Ensemble,
EBQL, PSDQN, PQN) are flagged as absent from all six surveyed repos in
§VII.

---

## Modern-RL methods folded into the axis structure

These methods had *no slot* in the pre-remap six-category framing;
the eight-axis structure gives them named homes. They are now part of
the full ~50-method method-type index in **supplement S2**, with the
canonical few per axis treated in the §IV body.

| Method | Year | Axis (§IV) | Why it matters |
|---|---|---|---|
| Maximin Q-Learning | 2020 | **W1 / §IV.A** | Generalization of double-Q to k-estimator min; rounds out the overestimation family |
| REDQ | 2021 | **W1 / §IV.A** | Random ensemble distillation — under-estimation pushback; the bias-variance trade-off made explicit |
| HER (Hindsight Experience Replay) | 2017 | **W2 / §IV.B** | Sparse-reward sample efficiency via goal relabeling |
| RND (Random Network Distillation) | 2018 | **W3 / §IV.C** | Curiosity-driven exploration; current go-to for Montezuma-class games |
| Go-Explore | 2019/2021 | **W3 / §IV.C** | Archival exploration; settles the Montezuma debate |
| CQL (Conservative Q-Learning) | 2020 | **W5 / §IV.E** | Canonical offline-RL Q-method |
| IQL (Implicit Q-Learning) | 2021 | **W5 / §IV.E** | Avoids OOD actions via expectile regression |
| BCQ (Batch-Constrained Q) | 2019 | **W5 / §IV.E** | The pre-CQL approach; explicit OOD handling |
| EDAC | 2021 | **W5 / §IV.E** | Ensemble + diversification penalty; bridges offline RL to ensemble Q |
| AWAC | 2020 | **W5 / §IV.E** | Advantage-weighted offline-to-online |
| VDN (Value Decomposition Networks) | 2018 | **W6 / §IV.F** | Additive Q decomposition for cooperative MARL |
| QMIX | 2018 | **W6 / §IV.F** | Monotonic mixing — the dominant value-decomp method |
| QPLEX | 2020 | **W6 / §IV.F** | Duplex dueling extension of QMIX |
| QTRAN | 2019 | **W6 / §IV.F** | Removes monotonicity constraint |
| Ape-X | 2018 | **W7 / §IV.G** | Distributed prioritized replay — the W7a throughput story |
| R2D2 | 2019 | **W7 / §IV.G** | Recurrent distributed replay |
| Agent57 | 2020 | **W7 / §IV.G** | Bandit-controlled exploration meta-policy; first to beat human on all 57 Atari games |
| MetaQ / MAML on Q | 2017–2020 | **W7 / §IV.G** | Meta-RL on value learning (W7b adaptation) |
| Munchausen DQN | 2020 | **W8 / §IV.H** | Add log-policy bonus — surprisingly strong, gets at stability via implicit KL regularization |
| Layer Norm + Residual variants (Smith et al. 2022/23) | 2022–23 | **W8 / §IV.H** | Recent stability recipes that PQN partially adopts |

Not all need full per-paper treatment — some get a one-paragraph
family treatment ("CQL, IQL, BCQ — three approaches to the OOD-action
problem; for full treatments see [refs]"). The §IV body treats the
canonical few per axis (§IV.E: CQL/IQL/BCQ; §IV.F: VDN/QMIX/QPLEX +
QFIX/Q+FIX; §IV.G: Ape-X/R2D2/Agent57); the complete enumeration lives
in supplement S2. Full continuous-action-space expansion
(DDPG/NAF/QT-Opt/CAQL/CQSM) was DECLINED as out of core scope for a
discrete-focused Q-learning survey; the hybrid PDQN is retained in
§IV.A.

---

## Cross-section appearances (the synthesis)

Methods that appear in multiple axis-sections, with cross-references,
are where the reviewers' "need conceptual insight" complaint gets
answered most directly. These are now consolidated in the §IV
**cross-axis interaction table** (W1–W8: origin / principal
interaction / deployment failure mode) — the one genuinely new
analytical idea adopted into core. The richest examples:

- **Rainbow** appears in W2/§IV.B (primary — PER and multi-step drive
  the ablation), and is cross-referenced from W1/§IV.A (Double Q
  component), W3/§IV.C (NoisyNet component), W4/§IV.D (distributional
  component), and W8/§IV.H (target network and dueling components).
  Rainbow is the *paradigm case* of methodological synthesis.
- **PQN** appears in W8/§IV.H (primary — its stability recipe is the
  contribution) and W7/§IV.G (its parallelism / vectorization). Its
  fundamental claim — "remove target networks and replay" — is a
  *stability* claim, not a category of its own.
- **SICQL / ICQL** sit in W7/§IV.G (scaling/adaptation) and are also
  referenced in §IV.J (foundation-model alignment, alongside
  Q-Transformer, VLM-Q, ShiQ, Q♯, Q-shaping).
- **Bootstrapped DQN / EBQL / UCB Q-Ensemble** all sit in W3/§IV.C as
  *ensembles for exploration*, with EBQL cross-referenced from
  W1/§IV.A as *ensembles for bias reduction*. The three are not just
  "Ensemble methods" — they target different axes with the same
  mechanism.
- **Distributional methods (C51 / QR-DQN / IQN / FQF)** sit in
  W4/§IV.D (richer credit-assignment signal) with explicit
  cross-reference to W1/§IV.A (the uncertainty/over-estimation
  interpretation). We pick W4 as load-bearing because the dominant
  empirical claim — that distributional methods *outperform* Rainbow
  without its other components — is a *signal* claim, not an
  *uncertainty* claim.

These cross-references are themselves a contribution, now made
explicit in the cross-axis interaction table rather than buried in
prose.

---

## What this doesn't tell us

This remap is now REALIZED in the final draft, not just a plan. It
does not by itself validate that:

- The §IV.E/F/G sections (offline / MARL / scaling) hold journal-
  quality depth — these were written; math depth (DDQL reciprocal-
  bootstrapping, Q+FIX/Dec-POMDP, SICQL/ICQL losses, PQN LayerNorm-
  Lipschitz, scaling-law formulas) was ROUTED to supplement S5 to
  respect the 21-page cap
- Every cross-reference resolves cleanly without prose contortions
  (the §IV cross-axis interaction table is the consolidated check)
- The problem-first taxonomy claim is defensible — confirmed by the
  2024–2026 prior-art sweep in `07-prior-art-sweep.md`; no competing
  problem-first organization found. "First" language hedged to "to our
  knowledge."

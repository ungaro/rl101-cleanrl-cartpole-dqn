# Outline — Problem-First Q-Learning Survey (TAI submission)

**Synced to v0.24 (2026-05-30).** This is THE OUTLINE — it tracks the
*realized* structure of `draft-tai/`, not the pre-distillation plan.
Status: submission-ready. **19-page TAI paper** (two-column IEEEtran,
double-anonymous, `\anontrue`) **+ 10-page supplement** (S1–S6).
Comprehensive ~47-page single-column source frozen as `draft-monograph/`
(git tag `monograph-v0.15`) — the "quarry" the paper distills from.

Governing principle of the distillation arc (47 → 34 → 26 → 27 → 21 → 19
pp): *"distill into a lens, don't delete."* Cut material lives in the
monograph and is curated into the supplement, never discarded.

The §IV top-level sections are *axes* — weaknesses of vanilla Q-learning
that a family of methods addresses — not *method types*. The method-type
taxonomy survives only as the analytic spine defined once in §IV's opening
table and indexed in full in supplement S2.

---

## Main paper — `draft-tai/` (≈19 pp, two-column)

### Front matter (compressed)
Title (12 words) · abstract (179 words, ≤250) · impact statement
(135 words, 100–150) · 5 keywords (TAI dropdown). No acknowledgements /
funding under `\anontrue`. Author list (Colby Wang, Ti, Divya, Kevin,
Hamna, Logan, Eason Yishan Wu, Charles Jiahao Zhang, Alp Guneysel)
preserved in the `\else` branch for camera-ready only.

### §I Introduction — *framework-first* (~1 pp)
Problem-first reframe: the field is not parallel inventions but eight
running attempts to fix eight weaknesses of vanilla Q-learning. Five
explicit contributions. Pre-announces the eight axes.

### §II Background (~1.5 pp)
MDP / Q-learning formalism, then eight weaknesses **W1–W8**, one sentence
each. **W7 is a COMPOSITE axis**: W7a sample throughput + W7b slow
adaptation. These eight names are the spine the §IV axis-sections answer.

| | Weakness |
|---|---|
| W1 | Overestimation bias (`max` over noisy estimates) |
| W2 | Sample inefficiency (uniform replay) |
| W3 | Brittle exploration (ε-greedy is dithered, not directed) |
| W4 | Reward sparsity & credit assignment |
| W5 | Distribution shift (offline RL) |
| W6 | Multi-agent coordination |
| W7 | **Composite:** W7a throughput / scaling + W7b slow adaptation |
| W8 | Function-approximation instability (deadly triad) |

### §III Methodology — systematic / PRISMA (~1.5 pp)
Explicit **systematic/PRISMA** protocol: databases (Google Scholar, arXiv
cs.LG/cs.AI, Semantic Scholar), search strings, 5 inclusion/exclusion
criteria, screening funnel ~200 → 120 → 80. **Table I** = prior-survey
comparison (six dimensions vs. five prior surveys). Full search log and
prior-art overlap → supplement S1.

### §IV "Q-Learning Methods by Weakness" (~8–9 pp — the core)
RETITLED from "Related Works." Terminology standardized to **"method-type
taxonomy"** — the word *"legacy"* is KILLED throughout.

**Overview / front apparatus:**
- **Method-type taxonomy table** up front — six categories → which axes
  (defined once here; full ~50-method index → supplement S2).
- **Genealogy figure** (master) + **branches figure** (modern-RL subgraph).
- **Axis × mechanism matrix.**
- **Cross-axis interaction table** — W1–W8 origin / principal interaction /
  deployment failure mode. (The one genuinely new analytical idea adopted
  from the LLM reviews.)

**§IV.A–H — uniform compact template per axis** (run-in **bold** lead-ins,
NO lettered subsubsections):
> *Weakness → Mechanisms (families by what they exploit) → Trade-off
> (the analysis) → Open questions → one comparison table.*

| Sub | Axis (weakness) |
|---|---|
| §IV.A | Overestimation bias (W1) — Double Q/DDQN, Dueling, Rainbow, EBQL, Maximin Q; bias↔variance, PDQN hybrid retained |
| §IV.B | Sample inefficiency (W2) — PER, MeDQN, DQfD, HER |
| §IV.C | Brittle exploration (W3) — NoisyNet, Param-Space Noise, Bootstrapped DQN, UCB Q-Ensemble, CBDQ, RND |
| §IV.D | Reward sparsity & credit assignment (W4) — C51, QR-DQN, IQN, FQF, multi-step / λ-returns (distributional = richer credit signal) |
| §IV.E | Distribution shift / offline RL (W5) — CQL, IQL, BCQ, BRAC, EDAC, AWAC; D4RL not Atari |
| §IV.F | Multi-agent coordination (W6) — VDN, QMIX, QPLEX, QTRAN; SMAC |
| §IV.G | Scaling & slow adaptation (W7 composite) — Ape-X, R2D2, Agent57, Meta-Q/MAML-Q, DRQN, PQN |
| §IV.H | Function-approximation instability (W8) — target net, Polyak, LayerNorm, PQN recipe, MeDQN consolidation, Munchausen |

**§IV.I Theoretical synthesis** (~250 words) — meta-section over the eight
axes: tabular convergence, deadly-triad counterexamples, distributional
contraction, finite-time bounds, offline pessimism, QPLEX IGM. Proofs and
derivations → supplement S5.

**§IV.J Foundation-model alignment** (~336 words) — framed as an EMERGING
direction: Q-Transformer, VLM-Q, ShiQ, Q♯, SICQL/ICQL, Q-shaping.

### §V Atari — diagnostic synthesis (~1.5 pp)
DIAGNOSTIC framing: task-category × axis. rliable / point-estimate caveat
kept. Dashes reframed as *evidence* (a method not reporting Montezuma
cannot claim to solve exploration). Full per-game tables → supplement S3.

### §VI Tabular — reproducible experiment (~1.5 pp)
NEW reproducible experiment (`scripts/tabular_experiments.py` +
`data/tabular_results.json`): Q-learning / SARSA / Expected SARSA /
3-step Q over **100 seeds + 95% bootstrap CIs** on FrozenLake / Taxi /
CliffWalking. **VI/PI/MPI/CVPI reported as a SEPARATED planning oracle**
(upper bound, full model access — NOT a model-free competitor).
Headline results: FrozenLake Q 0.722 / oracle 0.739; Taxi ≈7.93 /
oracle 7.935; CliffWalking Q −13 (optimal) / SARSA −79.9 (safe, high
variance) / Expected SARSA −17 / oracle −13. Full results (±std, CIs) →
supplement S6.

### §VII Repositories (~1 pp)
Six repos (Tianshou, XuanCe, CleanRL, DQN Zoo, SB3, RLlib) annotated by
axis. **Nine methods absent from ALL six**: DRQN, CBDQ, DQfD, MeDQN,
Bootstrapped DQN, UCB Q-Ensemble, EBQL, PSDQN, PQN. Hundal et al.
non-interchangeability point; named-vs-feature-equivalent caveat; per-repo
trade-off table. Full coverage matrix + design table → supplement S4.

### §VIII Conclusion (~0.5–1 pp)
Per-axis summary (now folds in §IV.I/J); community-repository proposal;
open directions; limitations & ethics (benchmark monoculture, compute
inequality, citation bias). "First" language hedged to "to our knowledge."

---

## Supplement — `draft-tai/supplement.pdf` (≈10 pp, S1–S6)

| § | Contents |
|---|---|
| S1 | Systematic search log + prior-art overlap |
| S2 | Full ~50-method **method-type index** (3 tables) — *was Appendix A* |
| S3 | Full Atari per-game tables — *moved out of §V* |
| S4 | Full repository coverage matrix + design table — *moved out of §VII* |
| S5 | Notation + 7 derivations/proofs — *was Appendix B* |
| S6 | Tabular-experiment config + full results (±std, CIs) — *backs §VI* |

---

## What MOVED to the supplement
- **Full Atari per-game tables** → S3 (§V keeps only the diagnostic
  task-category × axis synthesis).
- **Method-type index** (old Appendix A "legacy indexer", ~50 methods,
  3 tables) → S2.
- **Notation + proofs/derivations** (old Appendix B) → S5; §IV.I points here.
- **Full repository coverage matrix** → S4 (§VII keeps the per-repo
  trade-off table + the nine-absent-methods finding).
- **Systematic search log** → S1 (§III keeps the PRISMA prose + Table I).

## What was CUT or DEMOTED
- **§IV.I/J demoted** from full sections to a ~250-word theory synthesis
  and a ~336-word alignment subsection (proofs offloaded to S5).
- **"Legacy" terminology KILLED** — no longer a dual-view organizing
  device; the method-type taxonomy appears once (§IV table) and in S2.
- **Separate "Foundations" section dropped** — foundational methods stay
  in their axis-sections (multi-step in §IV.D, PQN in §IV.H, tabular
  baselines as §VI cross-refs); a standalone section would duplicate.
- **Continuous-action expansion DECLINED** (DDPG/NAF/QT-Opt/CAQL/CQSM) —
  out of core scope for a discrete-focused survey at the 21-page cap;
  only the PDQN discrete-continuous hybrid retained (§IV.A).
- **Added math depth ROUTED to supplement, not core** (per LLM-review
  triage): DDQL reciprocal-bootstrapping eqs, Q+FIX formula + Dec-POMDP
  V(h,s), SICQL/ICQL losses, PQN LayerNorm-Lipschitz contraction,
  scaling-law formulas. TAI explicitly discourages over-use of math.
- **Atari "reporting gaps limit fair ranking" apologia dropped** — the
  dashes are now evidence, not an excuse.

---

## Reviewer-triage outcome (v0.14 LLM reviews)
- **Adopted into core:** cross-axis interaction table (§IV); planning-oracle
  separation (§VI); 5 → 100 seeds + bootstrap CIs (§VI); systematic/PRISMA
  framing (§III).
- **Already present, not "missing":** planning-oracle framing, Hundal
  non-interchangeability, rliable caveat, CBDQ/QFIX/SICQL/ShiQ/Q♯.
- **Routed to supplement:** math depth (see CUT/DEMOTED above).
- **Declined/bounded:** full continuous-action expansion.

## Build & hygiene
- `build-tai.sh` → `main.pdf`; `build-supp.sh` → `supplement.pdf`.
  pandoc → LaTeX fragment, `--natbib` (IEEE `[N]` via `IEEEtran.bst`) +
  `tables-twocol.lua` (longtable → table*); system xelatex/bibtex;
  `refs.bib` (~136 entries).
- All automated checks PASS: `\anontrue`; no author/identity/internal-notes
  leakage in either PDF; two-column IEEEtran; abstract 179 w / title 12 w /
  impact 135 w / 5 keywords.
- **USER-SIDE TODO:** pick keywords from TAI dropdown; iThenticate
  similarity (≤20%); ORCID for all authors; flip `\anonfalse` only for
  camera-ready.

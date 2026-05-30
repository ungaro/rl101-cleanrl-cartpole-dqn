# Section Author Notes

Per-section material notes for LaTeX integration. Bibliography
additions, citation placement, scope decisions, and cross-references
that are useful to keep track of but do not belong in the paper body.

Organized by section.

> **Synced to v0.24 (2026-05-30).** STATUS: all core sections (§I–§VIII)
> distilled into draft-tai; submission-ready. Main 19pp two-column
> IEEEtran + 10pp supplement (S1–S6). Cut/extended material lives in
> `draft-monograph/` (frozen, tag `monograph-v0.15`) and is curated into
> the supplement. Governing principle held throughout: "distill into a
> lens, don't delete." Per-section status below.

---

## §I Introduction — DONE (distilled)

- **Status.** Reframed framework-first / problem-first; five
  contributions enumerated. In core.
- **Citation placement.** Ghasemi et al. 2024/2025 (arXiv:2411.18892)
  cited as the closest contemporary broad-RL survey, alongside the
  comparator surveys [12]–[15].
- **§VII positioning.** Single sentence distinguishing this paper's
  taxonomic repository analysis from Hundal et al. (2025)'s empirical
  PPO reproducibility audit; Hundal citation enters in §I, paid off in
  §VII. See `07-prior-art-sweep.md`.
- **Residual.** Front-matter compression already applied (the 21→19pp
  step). No TODO.

## §II Background — DONE (distilled)

- **Status.** MDP/Q-learning formalism + the eight weaknesses W1–W8,
  one sentence each. In core. NOTE: W7 is now a COMPOSITE axis —
  W7a (sample throughput) + W7b (slow adaptation); keep both clauses
  when citing W7.
- **Bibliography to verify.** Thrun & Schwartz 1993 and Sutton & Barto
  2018 (2nd ed., deadly-triad framing) cited in §II.B; confirm both in
  bibliography.
- **Notation.** The conventions subsection is shared by every
  subsequent section. Full notation table + proofs MOVED to supplement
  S5; core does not redefine
  $\theta, \theta^-, \alpha, \gamma, \pi, \mathcal{D}$.
- **Residual.** None.

## §III Methodology — DONE (distilled)

- **Status.** Now an explicit systematic/PRISMA protocol (databases:
  Google Scholar, arXiv cs.LG/cs.AI, Semantic Scholar; search strings;
  5 inclusion/exclusion criteria; screening ~200→120→80). Satisfies
  TAI's "explicit systematic methodology required." In core.
- **Table I.** Prior-survey comparison, six dimensions vs five prior
  surveys; Ghasemi et al. 2024/2025 (arXiv:2411.18892) included as a
  comparator. Check-mark pattern favorable (`07-prior-art-sweep.md`).
- **Moved to supplement.** Full systematic search log + prior-art
  overlap → S1.
- **Residual.** None.

## §IV "Q-Learning Methods by Weakness" — DONE (distilled, RETITLED)

- **Status.** Retitled from "Related Works." Opens by defining the
  **method-type taxonomy** once via a table (six categories → which
  axes). Terminology standardized to "method-type taxonomy" —
  "legacy" wording KILLED throughout. In core.
- **Overview visuals.** Genealogy figure, modern-RL branches figure,
  axis × mechanism matrix, and the **cross-axis interaction table**
  (W1–W8 origin / principal interaction / deployment failure mode —
  the one genuinely new analytical idea adopted from the LLM reviews).
- **Figure rendering.** The two mermaid figures (master genealogy +
  branches) are design renderings; final paper figures should be
  TikZ/PGF. Master genealogy should sit as a single landscape
  half-page figure, not two vertical ones.
- **Priority.** The axis × mechanism-family matrix is the most central
  visual to the thesis; if only one paper-level figure is rendered,
  prioritize this matrix.
- **Subsection template.** §IV.A–H use a UNIFORM COMPACT TEMPLATE:
  *Weakness → Mechanisms (families by what they exploit) → Trade-off →
  Open questions → one comparison table*, with run-in **bold**
  lead-ins (no lettered subsubsections).
- **§IV.I theoretical advances.** ~250-word synthesis in core; proofs
  → supplement S5.
- **§IV.J foundation-model alignment.** ~336 words, framed as an
  EMERGING direction (Q-Transformer, VLM-Q, ShiQ, Q♯, SICQL/ICQL,
  Q-shaping). In core.
- **Residual.** None — §IV.I/J demotion and the cross-axis table are
  the distillation steps that landed this section.

## §IV.B Sample Inefficiency

- **Bibliography.** HER: Andrychowicz et al. 2017, NeurIPS,
  arXiv:1707.01495.
- **Cross-references.** §IV.A (Rainbow), §IV.C (Private Eye
  exploration), §IV.E (offline RL sample-efficiency transfer),
  §IV.G (distributed training), §IV.H (MeDQN consolidation loss).

## §IV.C Brittle Exploration

- **Bibliography.** RND: Burda et al. 2018, arXiv:1810.12894.
  Go-Explore: Ecoffet et al. 2019/2021, *Nature* 2021.
- **Cross-references.** §IV.A (EBQL ensemble architecture shared),
  §IV.B (DQfD demonstration trade-off), §IV.D (distributional methods
  on the exploration diagonal), §IV.G (Agent57's bandit-controlled
  exploration meta-policy).

## §IV.D Reward Sparsity and Credit Assignment

- **Cross-references.** §IV.A (Rainbow's ablation evidence), §IV.B
  (PER as the other major Rainbow contributor), §IV.C (the
  exploration-versus-credit-assignment empirical separation), §IV.E
  (distributional methods in offline RL — EDAC, IQL).

## §IV.E Distribution Shift (Offline RL)

- **Bibliography (large set).**
  - BCQ: arXiv:1812.02900
  - BRAC: arXiv:1911.11361
  - AWAC: arXiv:2006.09359
  - CQL: arXiv:2006.04779
  - IQL: arXiv:2110.06169
  - EDAC: arXiv:2110.01548
  - D4RL: arXiv:2004.07219
  - Cal-QL: arXiv:2303.05479
  - Decision Transformer: arXiv:2106.01345 *(open-questions only)*
  - Gato: arXiv:2205.06175 *(open-questions only)*
- **Single-axis precedent.** Springer NCAA 2026 distribution-shift
  survey (10.1007/s00521-026-11966-8) cited at the head of this
  section as the existing single-axis problem-first precedent. See
  `07-prior-art-sweep.md`.
- **Scope decision.** Sequence-modeling approaches (Decision
  Transformer, Trajectory Transformer, Gato) are deliberately scoped
  out as they are not Q-learning; they appear in open-questions only.
- **Cross-references.** §IV.A (ensembles for bias control, EDAC),
  §IV.B (HER trajectory relabeling), §IV.H (stability mechanism
  interaction).

## §IV.F Multi-Agent Coordination

- **Bibliography.** VDN: arXiv:1706.05296; QMIX: arXiv:1803.11485;
  QPLEX: arXiv:2008.01062; QTRAN: arXiv:1905.05408; SMAC:
  arXiv:1902.04043.
- **Empirical numbers.** SMAC table drawn from QPLEX's reported
  numbers under the standard SMAC protocol. Independent replication —
  Hu et al. 2021 — reports different absolute numbers but consistent
  relative rankings.
- **Scope decision.** Competitive MARL (Nash Q-learning, opponent
  modeling) and mixed settings scoped out; scope is cooperative value
  decomposition.
- **Cross-references.** §IV.E (offline + multi-agent cross-axis open
  problem), §IV.H (function-approximation instability inheritance).

## §IV.G Scaling and Slow Adaptation

- **Note.** This subsection carries the COMPOSITE W7 axis (W7a sample
  throughput + W7b slow adaptation); keep both threads visible.
- **Bibliography.** Ape-X: arXiv:1803.00933; R2D2: ICLR 2019;
  Agent57: arXiv:2003.13350; NGU: arXiv:2002.06038; MAML:
  arXiv:1703.03400; DRQN: arXiv:1507.06527 *(may already be cited)*.
- **Scope decision.** IMPALA, SEED RL, Sample Factory, Podracer
  mentioned only at cross-reference; full distributed-RL coverage
  would expand the section beyond budget.
- **Cross-axis observation.** Agent57's composition of mechanisms from
  §IV.B (distributed replay), §IV.C (bandit-controlled exploration +
  NGU intrinsic motivation), and this section (slow adaptation) is the
  strongest empirical evidence for the problem-first organization.
- **Cross-references.** §IV.B (PER at distributed scale), §IV.C
  (exploration heterogeneity), §IV.H (PQN's synchronous parallelism).

## §IV.H Function-Approximation Instability

- **Bibliography.** Munchausen DQN: Vieillard et al. 2020, NeurIPS,
  arXiv:2007.14430.
- **Cross-references.** §IV.A (Dueling's overestimation aspect),
  §IV.B (MeDQN as consolidation; PQN's replay-replacement argument),
  §IV.D ($n$-step interaction with target staleness), §IV.E
  (offline-RL stability mechanisms), §IV.G (PQN's parallelism).

## §V Atari Benchmark Analysis — DONE (distilled)

- **Status.** Recast as a DIAGNOSTIC synthesis (task-category × axis);
  rliable / point-estimate caveat kept. In core.
- **Moved to supplement.** Full per-game tables → S3.
- **Optional figure.** F-B2 (axis-stratified Atari grouped bars, per
  `08-figure-proposals.md`) sits naturally between §V.A and §V.B.
  Optional; not blocking submission.
- **Residual.** None blocking.

## §VI Tabular Empirical Evaluation — DONE (distilled, RE-RUN)

- **Status.** NEW reproducible experiment, in core. Q-learning / SARSA
  / Expected SARSA / 3-step Q over **100 seeds + 95% bootstrap CIs**
  on FrozenLake/Taxi/CliffWalking (up from 5 seeds — an adopted review
  item). VI/PI/MPI/CVPI reported as a SEPARATED **planning oracle**
  (upper bound, full model access — explicitly NOT a model-free
  competitor).
- **Key numbers.** FrozenLake Q 0.722 / oracle 0.739; Taxi ≈7.93 /
  oracle 7.935; CliffWalking Q −13 (optimal) / SARSA −79.9 (safe,
  high variance) / Expected SARSA −17 / oracle −13.
- **Reproducibility.** `scripts/tabular_experiments.py` +
  `data/tabular_results.json`; config + full results (±std, CIs) →
  supplement S6.
- **Methodological motivation.** "Isolating algorithmic design from
  architectural confound" surfaced in abstract/§I.
- **Residual / declined.** FQE offline-RL tabular baseline was an
  optional expansion — NOT added; out of budget. The 100-seed +
  oracle-separation rework is the distillation that closed this
  section.

## §VII Repository Comparison — DONE (distilled)

- **Status.** Six repos (Tianshou, XuanCe, CleanRL, DQN Zoo, SB3,
  RLlib) compared by axis. In core. Includes the **nine methods absent
  from ALL six** (DRQN, CBDQ, DQfD, MeDQN, Bootstrapped DQN, UCB
  Q-Ensemble, EBQL, PSDQN, PQN); Hundal et al. non-interchangeability;
  named-vs-feature-equivalent caveat; per-repo trade-off table.
- **Moved to supplement.** Full coverage matrix + design table → S4.
- **Roadmap for §VIII.** The "absent from all six" list is dual-use:
  evidence for the implementation gap here, and the first-tranche
  roadmap for the community repository proposed in §VIII.B.
- **Residual.** None.

## §VIII Conclusion — DONE (distilled)

- **Status.** Per-axis summary (now folds in §IV.I theoretical
  advances + §IV.J foundation-model alignment); community-repository
  proposal; open directions; limitations & ethics (benchmark
  monoculture, compute inequality, citation bias). In core. No
  acknowledgements/funding (double-anonymous).
- **Repository spin-off URL.** If a Q-learning–specific repository is
  stood up before submission (cf. `01-pitch-analysis.md` Suggestion
  C), cite the URL inline in §VIII.B; otherwise the future-work
  commitment suffices.
- **Residual.** None.

---

## Supplement (S1–S6) — DONE (curated from monograph)

Separate 10-page document; `build-supp.sh` → `supplement.pdf`. All
"supplementary material" references from the main text resolve.

- **S1 — Systematic search log + prior-art overlap.** Backs §III
  PRISMA protocol. Done.
- **S2 — Full ~50-method method-type index (3 tables).** Formerly
  Appendix A (the method-type index / ex-"legacy indexer"). MOVED here
  from core. The reverse view (each category's distribution across the
  eight axes) is the analytical payload; the forward view is
  reference. Done.
- **S3 — Full Atari per-game tables.** Backs §V diagnostic synthesis.
  Done.
- **S4 — Full repository coverage matrix + design table.** Backs §VII.
  Done.
- **S5 — Notation + 7 derivations/proofs.** Formerly Appendix B; backs
  §II notation and §IV.I theoretical advances. Math-depth review items
  routed here (DDQL reciprocal-bootstrapping, Q+FIX + Dec-POMDP
  V(h,s), SICQL/ICQL losses, PQN LayerNorm-Lipschitz contraction,
  scaling-law formulas) rather than into core. Done.
- **S6 — Tabular-experiment config + full results (±std, CIs).** Backs
  §VI; pairs with `scripts/tabular_experiments.py` and
  `data/tabular_results.json`. Done.

---

## Review-triage outcomes affecting section notes

Three LLM reviews of v0.14 triaged. For the record, so per-section
edits are not re-litigated:

- **ADOPTED into core:** cross-axis interaction table (§IV overview);
  planning-oracle separation (§VI); 5→100 seeds + bootstrap CIs (§VI);
  systematic/PRISMA framing (§III).
- **ROUTED TO SUPPLEMENT:** all added math depth → S5 (see above).
- **DECLINED/BOUNDED:** full continuous-action expansion
  (DDPG/NAF/QT-Opt/CAQL/CQSM) out of scope for a discrete-focused
  survey; hybrid discrete-continuous (PDQN) retained in §IV.A. "First"
  language hedged to "to our knowledge."
- **Already present** (flagged as "must add" but in draft): planning
  oracle §VI, Hundal §VII, rliable §V, CBDQ/QFIX/SICQL/ShiQ/Q♯.

## User-side TODO (not section edits)

Tracked here only so they are not mistaken for open section work:
keywords from the TAI dropdown; iThenticate similarity (<20%); ORCID
for all authors; flip `\anonfalse` for camera-ready only.

# Q-Learning Survey — Direction Analysis

> **Synced to v0.24 (2026-05-30).**
> **Status:** the direction analyzed below was *realized*. The
> problem-first reframe (Suggestion A) became the spine of the paper;
> the genealogy figure (Suggestion B) and the repo proposal
> (Suggestion C) both landed. Venue is locked to **IEEE Transactions
> on Artificial Intelligence (TAI)**, "Original Research Review
> Manuscript." The paper is **submission-ready**: a 19-page
> two-column IEEEtran main paper (`draft-tai/`) plus a 10-page
> supplement (S1–S6). The comprehensive ~47-page single-column
> version is frozen at git tag `monograph-v0.15` (`draft-monograph/`)
> and now serves as the "quarry" for supplement material. This file
> is the *direction* record (why the reframe); the build/structure
> ground truth lives in the v0.24 state brief and `00-README.md`.
>
> Reading note: the reasoning below is preserved because it remains
> the correct rationale for the reframe. Where the original text
> speculated about open choices that are now decided (taxonomy
> terminology, two-column layout, references, author block, the
> figure rendering, the repo spin-off), those are marked **[RESOLVED
> v0.24]** inline rather than deleted.

## The core read of the reviewer feedback

The five reviewer points (algorithms feel isolated; need conceptual
insight; Atari is limited; too much derivation; need more modern RL)
all rhyme with **one structural complaint**:

> *The paper is organized as a catalogue. We want analysis.*

Every individual feedback point is a symptom of the catalogue
organization, not an independent problem:

| Surface complaint | Root cause |
|---|---|
| "Algorithms feel isolated" | Methods are presented in independent buckets with no axis of comparison |
| "Need conceptual insight" | The catalogue structure doesn't surface *why* a method exists |
| "Atari isn't enough" | Benchmark section is descriptive (what scores were obtained) rather than analytical (what does Atari fail to test) |
| "Too much derivation" | Derivation fills space that synthesis should occupy |
| "Need more modern RL" | The six-category taxonomy is built around 1989–2018 methods; modern RL doesn't fit the categories cleanly |

Fixing surface complaints one by one will produce a *bigger* catalogue.
Fixing the root cause means **reorganizing around a different axis** —
and once the axis is right, the surface complaints largely resolve
themselves.

---

## How the direction was realized (added v0.24)

The reframe above survived contact with a hard venue constraint, and
the *way* it survived is the real lesson. The governing principle that
drove every cut and every reorganization:

> **Distill into a lens, don't delete.** The contribution is the
> *framework* — the eight-weakness taxonomy and the analytical
> apparatus around it. The methods are *evidence* for the framework,
> not the point. Whenever the page budget bit, the question was never
> "which method do we drop?" but "which method's prose can be
> compressed into the lens without losing the analysis?" Cut material
> didn't vanish — it stayed in `draft-monograph/` (frozen at
> `monograph-v0.15`) and was curated forward into the 10-page
> supplement.

### The venue decision and what it forced

Locking to **IEEE TAI** ("Original Research Review Manuscript")
imposed a **21-page hard cap** (15 normal, mandatory \$200/page over
15), two-column IEEEtran, double-anonymous review, and an *explicit
systematic-methodology requirement*. That cap is what converted the
abstract "distill into a lens" principle into concrete edits. The
distillation arc by page count:

`47 → 34` (axis-sections §IV.A–H tightened) `→ 26` (§IV.I/J demoted,
appendices moved to supplement) `→ 27` (cross-axis interaction table
added back as net-new analysis) `→ 21` (§V/§VII distilled, §VI
re-run) `→ 19` (front-matter compression).

The end state is two editions plus a supplement:
- `draft-monograph/` — the ~47-page single-column "quarry," **frozen**.
- `draft-tai/` — the **19-page** two-column submission (`\ifanon`
  toggle; `\anontrue` for submission, named author list preserved for
  camera-ready).
- `draft-tai/supplement.pdf` — the **10-page** supplement (S1–S6),
  where the math depth, full Atari per-game tables, full repository
  matrix, ~50-method index, and proofs live.

So the TAI cap is what *forced* the compression from a monograph into
a tight paper + supplement — exactly the discipline the "lens"
principle prescribes, but with a number attached.

### The key framing decisions that made the lens hold

1. **Method-type taxonomy defined once, up front.** The six
   method-type categories (the old "botanist's" axis) were not
   deleted — they were demoted to a *single table at the head of §IV*
   ("Q-Learning Methods by Weakness," retitled from "Related Works")
   that maps each category to the axes it touches. Terminology was
   standardized to "method-type taxonomy"; the word **"legacy" was
   killed**. This is the resolution of the original tension between
   "pivot the axis" and "don't throw away the existing taxonomy": the
   taxonomy becomes a *reference grid*, the weaknesses become the
   *spine*. **[RESOLVED v0.24 — replaces the open "how locked is the
   taxonomy?" question below.]**

2. **Eight-axis uniform compact template.** Every axis subsection
   §IV.A–H follows the same shape: *Weakness → Mechanisms (families by
   what they exploit) → Trade-off (the analysis) → Open questions →
   one comparison table*, with run-in **bold** lead-ins instead of
   lettered subsubsections. This is the disciplined descendant of the
   five-part section sketch proposed in Suggestion A; the uniform
   template is what made eight axes fit in the page budget. (Note W7
   is a *composite* axis: W7a sample throughput + W7b slow adaptation,
   merged under the cap.)

3. **Cross-axis interaction table.** The one genuinely *new*
   analytical artifact — W1–W8 by origin / principal interaction /
   deployment failure mode. It is the payoff of the "physician's
   taxonomy" idea: it makes the weaknesses interact on the page
   instead of sitting in isolated buckets. It was the single item
   adopted into core from the (otherwise math-heavy, largely
   already-covered) LLM reviews of v0.14.

4. **§VI's reproducible experiment as original evidence.** The
   tabular section was upgraded from a descriptive table into a
   *first-party reproducible experiment*: Q-learning / SARSA /
   Expected SARSA / 3-step Q over **100 seeds with 95% bootstrap CIs**
   on FrozenLake / Taxi / CliffWalking, with VI/PI/MPI/CVPI reported
   as a **separated planning oracle** (an upper bound with full model
   access — explicitly *not* a model-free competitor). This is what
   lets a *survey* carry original evidence under a review-paper
   banner, and it directly answers the "too much derivation, not
   enough synthesis/evidence" complaint: the space derivations would
   have occupied now holds reproducible results
   (`scripts/tabular_experiments.py` + `data/tabular_results.json`).

Together these four decisions are how the abstract direction became a
submission-ready paper: the lens (weakness spine) is the contribution,
the methods are evidence, the cap forced compression rather than
deletion, and the supplement absorbed everything the lens didn't need.

---

## What the draft already does well

It's worth being precise about what's load-bearing in the current
version, because the structural pivot below preserves it rather than
discarding it.

- **"Critical Reflections" subsections already exist** at the end of
  each Related Works category. The bones of synthesis are there. The
  problem is the content — most reflections read as *open question
  prompts* ("future work could explore X") rather than substantive
  analysis, and they're trapped inside their method-type bucket so
  they cannot make cross-bucket comparisons.
- **The repository comparison (Tables VII/VIII) is genuinely useful.**
  Table VII's coverage matrix is exactly the kind of concrete
  synthesis the reviewers want; it just isn't *linked* to the method
  discussion.
- **Tabular benchmarks (Tables IV–VI)** are real first-party
  contributions and should stay where they are structurally.

The structural pivot below is an *upgrade* to the synthesis layer, not
a rewrite of the empirical layer.

---

## The modern-RL gap is structural, not cosmetic

Reviewer point 5 ("need more modern RL") looks like a small ask. The
draft makes clear it isn't:

| Modern Q-learning area | In the draft? |
|---|---|
| Offline RL (CQL, IQL, BCQ, EDAC) | **No** |
| Multi-agent (QMIX, VDN, QPLEX) | **No** |
| Distributed scale (Ape-X, R2D2, Agent57) | **No** |
| Meta-RL / continual Q-learning | **No** |
| Hindsight Experience Replay (HER) | **No** |
| World-models with Q-targets (MuZero family) | **No** |

The newest non-2025 deep-RL entries are MeDQN (2023) and PSDQN (2023).
Post-2018 deep-RL coverage is essentially two papers. CBDQ (2025) and
PQN (2025) round out the modernity story, but neither represents the
directions the field actually moved.

"Add a Modern RL appendix" is not enough — the current taxonomy has no
slot where offline RL or multi-agent value decomposition belongs.
This is what makes the structural pivot below *load-bearing* rather
than ornamental: those families naturally land in *distribution shift*
and *coordination* buckets that the current taxonomy doesn't admit.

---

## Suggestion A (high leverage, structural): pivot the organizing axis from method type to problem solved

The current taxonomy has six method-type buckets (Statistical,
Q-Function Computation, Memory/Replay, Ensemble, Model-Based, Minimal).
This is a *botanist's* taxonomy — it tells you what *kind* of thing
each method is. The reviewers want a *physician's* taxonomy — what
*symptom* does each method treat.

Proposed alternative organizing axis: **the foundational weaknesses of
vanilla Q-learning that modern methods address**. Sketch:

| Weakness of vanilla Q | Methods that target it |
|---|---|
| **Overestimation bias** (max operator + noisy estimates) | Double Q, Double DQN, Maximin Q, Weighted Double Q, EBQL, REDQ (under-est. pushback) |
| **Sample inefficiency** (replay uniformity, single trajectory) | PER, Rainbow, Ape-X, R2D2, HER, DQfD, MeDQN |
| **Brittle exploration** (ε-greedy is dithered) | NoisyNet, Bootstrapped DQN, IQN, UCB-ensembles, CBDQ, RND, Go-Explore |
| **Reward sparsity / credit assignment** | Distributional (C51, QR-DQN, IQN, FQF), n-step, λ-returns |
| **Distribution shift** (online → offline) | CQL, IQL, BCQ, BRAC, EDAC |
| **Coordination** (single-agent → multi-agent) | VDN, QMIX, QPLEX, QTRAN |
| **Slow adaptation** (per-task training) | Meta-Q (MAML on Q), Reptile-Q, distillation-based transfer |
| **Function-approximation instability** (the deadly triad) | Target networks, Polyak averaging, dueling decomposition, PQN's recipe |

Each section becomes:

1. **The weakness** — formal statement, why it's hard, what it costs.
2. **Solution families** — grouped by mechanism, compared head-to-head.
3. **Trade-offs introduced** — every fix has a cost; name it.
4. **Empirical evidence** — what ablation/benchmark data we have.
5. **Open question** — what's still unsolved in this dimension.

**Why this is the right pivot:**

- Directly addresses reviewer points 1 and 2 (synthesis + conceptual
  insight). The structure forces cross-method comparison — Double Q
  vs. Maximin vs. REDQ all share a section.
- Pulls modern RL into the existing structure rather than tacking it
  on. *Distribution shift* is the natural home for offline RL.
  *Coordination* is the home for multi-agent value decomposition. No
  "Modern RL" appendix needed.
- Addresses "Atari is limited" indirectly: a *problem-first* paper
  doesn't lean on Atari as the spine. The dashes in Tables II/III stop
  being apologies and start being evidence — Montezuma at 0 across the
  board *is* the brittle-exploration argument. Classic control becomes
  evidence for *stability*; offline benchmarks (D4RL) become evidence
  for *distribution shift*. The benchmark landscape becomes coherent.
- Justifies cutting derivations: a problem-first section is about the
  *mechanism* of the fix, not the formal derivation. Derivations move
  to a single appendix.

**Cost:** ~3–4 weeks of distributed work with 3+ contributors, gated
on lining up domain owners for offline RL, multi-agent, and distributed.

**Risk to flag:** if the draft is already in late revision, the cost
may be higher than the team can absorb in one cycle. The genealogy
figure in Suggestion B captures most of the synthesis benefit without
rewriting the prose.

---

## Suggestion B (medium leverage, visual): add a centerpiece *design-space* figure

If we cannot afford a full structural pivot, the next-highest-leverage
move is a **single well-designed figure that does the synthesis work**.
The reviewers complain about isolated methods because they have no
visual anchor for how methods relate. One good figure makes the
relationships visible at a glance.

Two candidates:

### B1. Q-learning *genealogy tree*

A directed graph of methods, with each edge annotated by the weakness
the child method addresses in the parent. Annotated edges *are* the
synthesis the reviewers want. The absence of CQL/IQL/QMIX appears as a
visible empty region in the figure — itself an argument for the
structural pivot.

(ASCII first pass in `06-genealogy-figure.md`.)

### B2. Q-learning *design-space scatter plot*

A 2D figure with axes like:
- x: sample efficiency (e.g. median DQN-normalized score at 10M frames)
- y: implementation complexity (or: number of components on top of vanilla DQN)
- color: family
- size: typical compute budget required

Each method is a point. The clusters make the relationships visible:
"Rainbow is high-complexity, high-efficiency; minimal Q-learning is
low-complexity, low-efficiency; ensembles are high-complexity,
medium-efficiency."

**Recommendation:** start with B1 (the genealogy). It's the cheaper of
the two, it doubles as the framing artifact for Suggestion A (the
discussion of "which edge label goes where" *is* the seed of the
structural pivot), and it makes the modern-RL gap visible.

---

## Suggestion C (low cost, high signal): spin off the Q-learning repo the Conclusion already promises

The current Conclusion contains:

> "Looking forward, a clear avenue for future research lies in the
> development of a dedicated, community-maintained repository focused
> exclusively on Q-learning and its deep variants."

This is a real spin-off opportunity. Even if the repo is just a
coverage-tracker plus stubs for the missing variants
(Bootstrapped DQN, EBQL, UCB Ensemble, PSDQN, MeDQN, CBDQ — *none* of
which are in any of the six surveyed repos per Table VII), the act of
building it converts the survey's findings into a community
deliverable. Costs ~1 day for the stub.

**First-week deliverable:** GitHub repo skeleton, README is Table VII,
tracking issues for the six absent variants with "wanted:
implementation" tags.

---

## Strategic recommendation

- **A (structural pivot) — done.** All eight axis-sections plus §I,
  §II, §III, §IV overview, §V–§VIII, and Appendix A are drafted in
  `draft/`. See `00-README.md` for the full status map.
- **B (genealogy figure) — done.** Design source in
  `06-genealogy-figure.md`; mermaid renderings in
  `draft/4-overview.md` (master genealogy + modern-RL subgraph) and
  `draft/4c-brittle-exploration.md` (exploration branch). Mermaid
  `quadrantChart` 2D positioning grids in §IV.A, §IV.C, §IV.D,
  §IV.E, §IV.H. Final TikZ/PGF rendering for the IEEE template is
  the only pending sub-item.
- **C (repo spin-off) — landed as a proposal in the paper.** Promoted
  to a named deliverable in §VIII (community-repository proposal +
  open directions) anchored on the **nine methods absent from all six
  surveyed repositories** (DRQN, CBDQ, DQfD, MeDQN, Bootstrapped DQN,
  UCB Q-Ensemble, EBQL, PSDQN, PQN). The standalone GitHub stub is
  out of scope for the submission itself and is not required for it.

---

## Open questions for the team before committing to a direction

**[RESOLVED v0.24]** — these were the gating questions before the
team committed. They are recorded as resolved:

1. **How locked is the taxonomy?** Resolved by *demotion, not
   deletion*: the six method-type categories now live in a single
   head-of-§IV table mapping categories→axes; the weakness spine is
   the contribution. "legacy" terminology killed. (See "How the
   direction was realized," decision 1.)
2. **Is there appetite for a co-author taking ownership of modern
   families?** Resolved: modern families are covered within the axis
   spine rather than as owned standalone sections — foundation-model
   alignment (Q-Transformer, VLM-Q, ShiQ, Q♯, SICQL/ICQL, Q-shaping)
   sits in §IV.J as an *emerging direction* (~336 words); theoretical
   advances in §IV.I (proofs→supp S5). Full continuous-action-space
   expansion (DDPG/NAF/QT-Opt/CAQL/CQSM) was **declined** as out of
   core scope for a discrete-focused survey at 21pp.
3. **What's the editor's tone?** Moot — the work was reframed as a
   fresh systematic review submission to TAI rather than a revision
   of a prior decision; §III now carries an explicit systematic/PRISMA
   protocol to satisfy TAI's methodology requirement.
4. **Should the repository comparison stay as a standalone section?**
   Resolved: kept as standalone §VII (six repos by axis + per-repo
   trade-off table; full matrix→supp S4), with the
   non-interchangeability and named-vs-feature-equivalent caveats
   that make it analytical rather than a sidebar.

---

## Differentiation check against 2024–2026 prior art

A focused web sweep (Google Scholar + arXiv cs.LG/cs.AI + Semantic
Scholar) was run to check whether other recent surveys threaten the
differentiation claims. Verdict: **the problem-first reorganization
angle is open.** No 2024–2026 Q-learning / DQN survey takes that
organizing structure; no paper unifies tabular + deep Q-learning into
a problem-axis framework. The structural pivot in Suggestion A is a
clean differentiation claim.

Three citations need to be added to the revision regardless of which
suggestion the team adopts:

- **Ghasemi et al. 2024/2025** (arXiv:2411.18892) — closest
  competitor, broad RL survey organized by method family. The
  proposed finer taxonomy + repo + Atari analysis differentiates.
- **Springer NCAA 2026** offline-RL distribution-shift survey
  (10.1007/s00521-026-11966-8) — single-axis problem-first precedent
  to cite at the head of §IV.E.
- **Hundal et al. 2025** (arXiv:2503.22575) — empirical
  reproducibility audit of RL libraries on PPO. Not a survey, but
  directly adjacent to our repository-comparison contribution.
  Requires one sentence in §VII distinguishing the angles.

Full findings, borderline cases, and follow-up checks in
`07-prior-art-sweep.md`.

---

*Companion files (all on `alp/q-survey-revision-prep`, not merged to main):*

- `03-new-outline.md` — problem-first table of contents
- `04-method-remap.md` — working notes for the legacy ↔ axis mapping; canonical version is `draft/A-legacy-indexer.md`
- `06-genealogy-figure.md` — genealogy figure design source (ASCII + mermaid)
- `07-prior-art-sweep.md` — 2024–2026 competing-surveys check
- `08-figure-proposals.md` — figure / comparison artifact catalogue
- `09-section-notes.md` — per-section material notes (bibliography, citations, scope decisions)
- `draft/` — the paper itself, one file per section

---

## Status snapshot (suggestion-by-suggestion)

- **Suggestion A — problem-first structural pivot:** Drafted in full.
  Nine §IV subsections (§IV.A–H axis-sections plus §IV.I theoretical
  advances), plus §I introduction, §II background + 8 weaknesses,
  §III methodology with Table I, §IV overview with master genealogy
  + axis × mechanism-family matrix, §V Atari benchmarks (now
  including §V.H limitations and §V.I newer benchmarks), §VI tabular
  empirical, §VII repositories, §VIII conclusion (with four
  cross-axis open directions including real-world distribution
  shift), Appendix A legacy indexer, Appendix B notation and
  selected derivations. Differentiation against 2024–2026 prior art
  confirmed in `07-prior-art-sweep.md`.
- **Suggestion B — centerpiece figure:** Genealogy figure design in
  `06-genealogy-figure.md`; mermaid renderings inserted into
  `draft/4-overview.md` (master genealogy + modern-RL branches) and
  `draft/4c-brittle-exploration.md` (exploration branch). Mermaid
  `quadrantChart` 2D positioning grids inserted in §IV.A, §IV.C,
  §IV.D, §IV.E, §IV.H. **[RESOLVED v0.24]** The §IV overview in
  `draft-tai/` ships the figures (genealogy, branches, axis×mechanism
  matrix) plus the **cross-axis interaction table**; rendering is
  resolved in the IEEEtran build.
- **Suggestion C — Q-learning repo spin-off:** Promoted to a named
  deliverable in §VIII (community-repository proposal) anchored on the
  nine methods absent from all six surveyed repositories. **[RESOLVED
  v0.24 — bounded]** The proposal is in the paper; standing up the
  external GitHub repo is out of scope for the submission.

## Reviewer-feedback coverage (after Sessions 1–3)

Full audit in `10-reviewer-audit.md`. All five reviewer-feedback
areas from the pitch deck now satisfy every sub-ask:

- **Feedback 1 (algorithms feel isolated)** ✓ — per-section
  comparison tables, 2D positioning grids, axis × mechanism-family
  matrix, genealogy figure, cross-references
- **Feedback 2 (need conceptual insight)** ✓ — eight-weakness spine,
  per-section "A. The Weakness", "C. Trade-offs"
- **Feedback 3 (Atari is not enough)** ✓ — §V.H limitations + §V.I
  newer benchmarks (Atari-100k, ALE-stochastic, ProcGen, NetHack,
  BSuite) + §VII.A reproducibility-crisis citations + §VIII.C
  real-world distribution-shift bullet (Session 2)
- **Feedback 4 (too much derivation)** ✓ — Appendix B notation +
  selected derivations; §I reading guide; §II.C notation conventions
  removes per-section redefinition burden (Session 3)
- **Feedback 5 (need more modern RL)** ✓ — §IV.E offline, §IV.F
  multi-agent, §IV.G distributed and meta (expanded with PEARL,
  ProMP, MQL, in-context Q-learning), §IV.I theoretical advances
  (Session 1)

## Outstanding items (independent of reviewer feedback)

**[RESOLVED v0.24]** — the build-side blockers below are all closed.
Recorded as resolved:

- **Bibliography / References section.** Resolved: IEEE `[N]`
  citations via `IEEEtran.bst` through pandoc `--natbib` + bibtex;
  `refs.bib` has ~136 entries.
- **Author block.** Resolved: `\ifanon` toggle, `\anontrue` set for
  the double-anonymous submission; named author list (Colby Wang, Ti,
  Divya, Kevin, Hamna, Logan, Eason Yishan Wu, Charles Jiahao Zhang,
  Alp Guneysel) preserved in the `\else` branch for camera-ready.
- **Two-column layout.** Resolved: switched to `documentclass=IEEEtran`;
  the longtable/twocolumn conflict is handled by `tables-twocol.lua`
  (longtable→`table*`). Builds via `build-tai.sh`→`main.pdf` and
  `build-supp.sh`→`supplement.pdf`.
- **Visual polish on tables and figures.** Resolved in the IEEEtran
  build; oversized tables routed to the supplement (S2/S3/S4).

**Still open (USER-SIDE, pre-submission):**

- Pick keywords from the TAI dropdown (3–6; currently 5 placeholders).
- Run iThenticate similarity check (≤20% required).
- ORCID for all authors.
- Flip `\anonfalse` only for camera-ready.
- *(Bounded, out of submission scope)* stand up the external
  Q-learning community repo (Suggestion C is a paper proposal, not a
  submission gate).

# TAI Compression Ledger

Working plan for distilling the comprehensive monograph (`draft/`,
preserved frozen at `draft-monograph/`, tag `monograph-v0.15`) into the
IEEE TAI edition (`draft-tai/`). Co-authors steer scope here before
prose is rewritten.

## Constraints (from TAI Information for Authors)

- **Review paper: 15 pages normal, 21 max** (two-column IEEEtran);
  $200/page over 15. Double-anonymous. Plagiarism ≤20%.
- Impact statement 100–150 words; abstract ≤250; title ≤15 words;
  3–6 keywords. Survey must have an explicit, systematic methodology
  (inclusion/exclusion filter) and synthesize into a new form.
- Culture rewards focus over enumeration ("compare against the top two
  or three competitive algorithms"); references should skew recent.

## Governing principle — distill, don't delete

The monograph is the **evidence base / extended version**; the TAI
paper is a **lens** whose contribution is the *framework*, with methods
cited as evidence for it. Cut material is not lost — it lives in
`draft-monograph/` and is curated into formal **Supplementary
Materials** later. Every section carries explicit *analysis* (trade-off
reasoning, what is settled vs. open), not just enumeration.

## Where the analysis lives (so the paper reads as analysis, not catalog)

- Per-axis **trade-off** + **open-question** paragraphs (kept; the
  enumerative method detail is what gets cut).
- §V: diagnostic synthesis — which axis-methods win which *task types*
  and why, with calibrated (not leaderboard) claims.
- §VI: one hypothesis-driven controlled experiment (20–30 seeds,
  bootstrap CIs, rliable) testing a taxonomy prediction.
- §VII: Pareto-frontier analysis of repository design trade-offs.
- Cross-cutting: the 8-axis taxonomy figure + legacy crosswalk +
  a compact "composite agents across axes" view (Rainbow, Agent57).

## Baseline

Monograph body = **36,338 words → 47 two-column pages.** Target ≤21.
Core-prose budget ≈ **10–11k words** + ~6 figures + ≤1 table/axis;
remainder → supplementary.

## Per-section disposition

| Monograph section | Words | TAI core target | Disposition |
|---|---|---|---|
| I. Introduction | 864 | ~700 | **Core.** Reframe framework-first; fix roadmap; rename §IV. |
| II. Background | 1293 | ~650 | **Core.** Only Bellman/DQN essentials + the 8-weakness statement. |
| III. Methodology | 1356 | ~700 + PRISMA fig | **Core, rebuilt** as a systematic protocol. Full ledger → supp. |
| IV. overview/taxonomy | 1301 | ~500 + taxonomy fig | **Core.** Becomes §IV preamble + the master taxonomy figure. |
| IV.A Overestimation | 2057 | ~550 + table | **Core (compact template).** Derivations/quadrant → supp. |
| IV.B Sample ineff. | 1398 | ~450 + table | Core (compact). |
| IV.C Exploration | 2085 | ~550 + table | Core (compact). |
| IV.D Reward/credit | 2035 | ~550 + table | Core (compact). |
| IV.E Distribution shift | 2285 | ~550 + table | Core (compact). |
| IV.F Multi-agent | 1740 | ~500 + table | Core (compact). |
| IV.G Scaling/adapt | 3203 | ~550 + table | Core (compact). Largest cut. |
| IV.H Stability | 1771 | ~500 + table | Core (compact). |
| IV.I Theoretical adv. | 1893 | ~250 | **Mostly supp.** Key results folded into axes + conclusion. |
| IV.J FM alignment | 1970 | ~350 | **Emerging-directions ½pp in core**; full → supp (TAI: frontier ≠ conclusive). |
| V. Atari benchmarks | 3380 | ~900 + 1–2 figs | **Core synthesis.** Full extraction tables → supp. |
| VI. Tabular empirical | 1276 | ~700 + fig | **Replace** with new hypothesis-driven experiment; extended → supp. |
| VII. Repositories | 1904 | ~700 + heatmap | **Core.** Full coverage matrix → supp; clarify 2013-vs-2015 DQN rows. |
| VIII. Conclusion | 1538 | ~600 | **Core.** Explicit, concise future work. |
| A. Legacy indexer | 1311 | crosswalk fig | **Supp** (full table); compact crosswalk figure in core. |
| B. Notation/proofs | 1678 | — | **Supp** entirely (TAI: proofs for grad students → supp). |

Core-prose subtotal ≈ **10.3k words** → ~21 pages with figures + refs.

## Per-axis compact template (uniform across §IV.A–H)

1. **Weakness.** 2–3 sentences; one mechanism-defining equation max.
2. **Mechanisms.** Solution families by *what they exploit*; one clause
   each; ≤2 equations total for the axis.
3. **Trade-off.** The analytical core — why no method dominates.
4. **Open question.** 1–2, concise.
5. **Comparison table** (wide → `table*`): Method | Mechanism | Cost |
   Best at | Empirical anchor. (Drops the monograph's "legacy category"
   column — that role moves to the Appendix A crosswalk.)

Heading fix: §IV becomes one `# IV.` section; axes become `## IV.x`
subsections (was one top-level section per axis). Within an axis,
run-in **bold** lead-ins replace `###` subsubsections to save space.

## Execution order

1. ✅ Ledger + §IV.A exemplar (this step) — agree the template.
2. §IV.B–H to the same template; rebuild, watch page count.
3. Front matter: Intro reframe, Background compress, Methodology→PRISMA.
4. §IV overview → taxonomy figure + preamble; §IV.I/J demote.
5. §V synthesis; §VII heatmap; §VIII conclusion.
6. §VI new experiment (separate workstream; 20–30 seeds + rliable).
7. Assemble Supplementary from `draft-monograph/`.
8. Final hygiene: anonymization sweep, ≤20% similarity, fonts/figures.

# Second-Round Reviewer Audit

> **Synced to v0.24 (2026-05-30).** Status: all major second-round asks
> resolved. Assessed against the distilled **19-page** two-column
> IEEEtran TAI draft (`draft-tai/`) + the **10-page** supplement
> (S1–S6). The reviewer thrust in this round was *systematic
> methodology + auditable scholarly apparatus + statistical rigor*; the
> distillation arc (47→19pp) plus the supplement absorbed essentially
> all of it. Per-ask resolution codes are recorded inline below
> (CLOSED / ADDRESSED-IN-SUPPLEMENT / DECLINED-WITH-RATIONALE /
> SUPERSEDED).
>
> **This is a DISTINCT round** from the three LLM reviews of v0.14,
> which are triaged separately in `14-llm-review-triage.md`. The two
> human-style reviewer reports below are the *second-round* feedback;
> do not conflate them with the LLM "add more math/algorithms" thrust.
>
> **Headline resolutions (v0.24):**
> - Systematic-methodology concern → **CLOSED** via §III explicit
>   PRISMA framing (databases, search strings, 5 incl/excl criteria,
>   screening ~200→120→80, Table I prior-survey comparison).
> - Auditable-artifacts concern → **CLOSED** via supplement S1–S6
>   (search log, full method-type index, per-game tables, repo coverage
>   matrix, proofs) + reproducible `scripts/tabular_experiments.py`.
> - Statistical-rigor concern → **CLOSED** via §VI re-run at **100
>   seeds + 95% bootstrap CIs** (was 5 seeds), planning-oracle
>   separated as an upper bound, not a peer.
> - Scope-control concern → achieved via **distillation** ("distill
>   into a lens, don't delete"): cut material lives in
>   `draft-monograph/` (FROZEN, tag `monograph-v0.15`) and is curated
>   into the supplement, not lost.
> - "First"/firstness language → **hedged** to "to our knowledge."

The original codes used in the per-ask tables below (pre-sync):

- ✓ already closed (often by Sessions 1–3 or the late-2025/2026 sweep)
- ⚙ quick fix (verifiable inconsistency, addressed in this turn)
- ⊘ new content gap (needs a focused session)
- ⊕ new structural concern (architectural, beyond a single session)

These have since been superseded by the v0.24 resolution codes; the
original audit is preserved below for the record, with each ask now
annotated **[v0.24: …]**.

Both reviewers keep the eight-axis problem-first framework as the
paper's central contribution — neither asks the team to abandon it.
The substantive asks are all *content additions* and *methodological
tightening*, not a structural retreat.

---

## Reviewer 1 — "Peer Review Report" (verdict: conditional accept, major revisions)

Reviewer 1 is broadly enthusiastic about the structural pivot and
treats the asks as additions to strengthen an already-publishable
draft. Their twelve substantive asks, mapped to the current draft:

| # | Ask | Status | v0.24 resolution | Note |
|---|---|---|---|---|
| 1 | QFIX superseding QPLEX for IGM-completeness | ✓ | **CLOSED** | Retained in §IV.F method-type taxonomy; QFIX formula → supp S5 |
| 2 | Cognitive Belief-Driven Q-Learning | ✓ | **CLOSED** | CBDQ in §IV.C; also in the nine-methods-absent §VII analysis |
| 3 | Plasticity loss / capacity crisis | ✓ | **CLOSED** | Covered in §IV.I synthesis + §IV.H; survey cites retained |
| 4 | Flow-matching policies with Q-learning | ✓ | **CLOSED** | FQL (Park, Li, Levine 2025) in §IV.E |
| 5 | Q-learning with Adjoint Matching | ⊘ | **DECLINED-WITH-RATIONALE** | Continuous adjoint method; out of core scope for a discrete-focused Q-survey at the page cap (same boundary as DDPG/NAF/QT-Opt). Quarried in `draft-monograph/` |
| 6 | Predictable scaling laws for value-based DRL (Rybkin et al.) | ⊘ | **ADDRESSED-IN-SUPPLEMENT** | §IV.G framing reconciled (W7a sample throughput vs W7b adaptation); scaling-law formulas routed to supplement, not core prose |
| 7 | Q-learning for LLM/VLM alignment (ShiQ, VLM Q-Learning, Q♯) | ⊘ | **CLOSED** | Now §IV.J foundation-model alignment (~336 words, framed as EMERGING): Q-Transformer, VLM-Q, ShiQ, Q♯, SICQL/ICQL, Q-shaping |
| 8 | Parameterized Q-Networks for hybrid discrete-continuous action spaces | ⊘ | **CLOSED** | PDQN retained in §IV.A (hybrid discrete-continuous) |
| 9 | Hamilton-Jacobi-Bellman continuous-time Q-learning | ⊘ | **DECLINED-WITH-RATIONALE** | Continuous-time; out of discrete-core scope at page cap. Quarried in monograph |
| 10 | Crossing-quantiles problem and non-crossing QR-DQN variants | ⊘ | **ADDRESSED-IN-SUPPLEMENT** | Distributional axis §IV.D distilled to the compact template; per-variant detail in supp |
| 11 | Transfer learning + feature-correlation regularization for overestimation | ⊘ | **DECLINED-WITH-RATIONALE** | Minor; folded into §IV.A overestimation trade-off rather than its own paragraph at 19pp |
| 12 | Branching parameterized Q-networks for compound overestimation | ⊘ | **SUPERSEDED** | Companion to #8; subsumed by the §IV.A PDQN treatment |

**Reviewer 1 verdict (original):** of twelve asks, four already done, one moderate (#10), seven new content additions; biggest single ask was #7 (LLM/VLM alignment). **[v0.24]** #7 is now CLOSED as §IV.J. Of the twelve: 5 CLOSED, 2 ADDRESSED-IN-SUPPLEMENT, 1 SUPERSEDED, 3 DECLINED-WITH-RATIONALE (continuous/minor items bounded out of the discrete-focused core and preserved in `draft-monograph/`).

---

## Reviewer 2 — "Reviewer Report" (verdict: major revision / R&R; lean reject if binary)

Reviewer 2 is more critical and focuses on scholarly apparatus, not
content. Their asks are organized into bibliographic, methodological,
and taxonomic categories.

### Bibliographic / scholarly apparatus (highest priority)

| Ask | Status | v0.24 resolution | Note |
|---|---|---|---|
| References list / bibliography missing | ⊕ | **CLOSED** | Bibliography now wired: pandoc `--natbib` + `refs.bib` (~136 entries), IEEE `[N]` via `IEEEtran.bst`, bibtex in `build-tai.sh` |
| Audit every citation, year, venue | ⊕ | **CLOSED** | Done against `refs.bib`; PQN re-dated (below) as part of the sweep |
| Fix internal count error ("Eight methods" vs nine listed) | ⚙ | **CLOSED** | §VII now states **nine** methods absent from all six repos (DRQN, CBDQ, DQfD, MeDQN, Bootstrapped DQN, UCB Q-Ensemble, EBQL, PSDQN, PQN) |
| PQN dating: 2025 vs 2024 | ⚙ | **CLOSED** | Gallici et al. arXiv:2407.04811 corrected to 2024 across draft + `refs.bib` |
| MDP notation $P(s' \mid s, a, \theta)$ — $\theta$ doesn't belong in environment kernel | ⚙ | **CLOSED** | §II MDP formalism corrected; $\theta$ removed from the environment kernel |

### Methodological transparency (PRISMA-style)

| Ask | Status | v0.24 resolution | Note |
|---|---|---|---|
| Search databases, queries, screening counts | ⊘ | **CLOSED** | §III now an explicit systematic/PRISMA protocol: databases (Google Scholar, arXiv cs.LG/cs.AI, Semantic Scholar), search strings, 5 incl/excl criteria, screening ~200→120→80; full search log → supp S1 |
| Date-stamped literature cutoff | ⊘ | **CLOSED** | Explicit cutoff stated in §III methodology |
| Axis-assignment protocol | ⊘ | **ADDRESSED-IN-SUPPLEMENT** | Per-method axis assignment is the supp S2 method-type index (~50 methods, 3 tables) |
| Per-paper extraction template | ⊘ | **ADDRESSED-IN-SUPPLEMENT** | Extraction captured in supp S1 search log + S2 index |
| Whether paper is narrative / scoping / systematic | ⊘ | **CLOSED** | §III explicitly frames the work as a **systematic** review (PRISMA protocol); resolves the framing question |

### Reproducibility of empirical sections

| Ask | Status | v0.24 resolution | Note |
|---|---|---|---|
| §VI tabular study under-specified | ⊕ | **CLOSED** | Re-run reproducible: `scripts/tabular_experiments.py` + `data/tabular_results.json`; **100 seeds + 95% bootstrap CIs** (was 5); full config + results (±std, CIs) → supp S6 |
| §VI baseline framing (VI/PI/MCTS as oracle vs peer) | ⊘ | **CLOSED** | VI/PI/MPI/CVPI now a SEPARATED **planning oracle** (upper bound, full model access — explicitly NOT a model-free competitor) |
| §VI define abbreviations (CVPI, MPI) on first use | ⚙ | **CLOSED** | Defined on first use in §VI |
| §VII repository audit needs version pins | ⊘ | **ADDRESSED-IN-SUPPLEMENT** | Full repo coverage matrix + design table → supp S4; per-repo trade-off table in core §VII |
| Distinguish named-algorithm support vs feature-equivalent support | ⊘ | **CLOSED** | §VII carries the named-vs-feature-equivalent caveat + Hundal et al. non-interchangeability point |
| Statistical reporting: confidence intervals, effect sizes, rliable engagement | ⊘ | **CLOSED** | §VI CIs (above); §V keeps the rliable/point-estimate caveat (Agarwal et al. 2021) |
| Castro et al. 2020 *Revisiting Rainbow* engagement | ⊘ | **CLOSED** | §V is a DIAGNOSTIC task-category × axis synthesis (axis-stratified evaluation precedent); cite in `refs.bib` |

### Taxonomic / framing concerns

| Ask | Status | v0.24 resolution | Note |
|---|---|---|---|
| Distributional RL W4 placement (credit assignment vs uncertainty) | ⊕ | **CLOSED** | Framed as an interpretive lens within the §IV.D compact template; cross-axis interaction table records W4 origin/interaction explicitly |
| W7 "Slow adaptation" too coarse (mixes distributed + recurrence + meta) | ⊘ | **CLOSED** | Adopted option (b): W7 explicitly relabeled a **COMPOSITE axis** — W7a sample throughput + W7b slow adaptation |
| Dueling DQN re-classification under §IV.H | ✓ | **CLOSED** | Defended via Rainbow ablation evidence; retained under §IV.H |
| "Claims of firstness" need backing | ✓ | **CLOSED** | "First"/firstness language hedged to "to our knowledge"; backed by the systematic search log (supp S1) + Table I prior-survey comparison |
| Engage Castro 2020 + Agarwal 2021 (eval methodology) | ⊘ | **CLOSED** | See reproducibility row above — both engaged in §V/§VI |

### Limitations and ethics

| Ask | Status | v0.24 resolution | Note |
|---|---|---|---|
| Dedicated limitations / ethics section | ⊘ | **CLOSED** | Consolidated in §VIII: limitations & ethics (benchmark monoculture, compute inequality, citation bias) |

**Reviewer 2 verdict (original):** of ~22 distinct asks, three quick fixes, three already done, two bibliography work, fourteen new methodological/content additions; harsher tone reflected the scholarly-apparatus focus. **[v0.24]** All four pillars Reviewer 2 leaned on are now closed: bibliography wired (`refs.bib` + `--natbib`), systematic methodology (§III PRISMA), statistical rigor (§VI 100 seeds + CIs, planning-oracle separation), and auditable artifacts (supp S1–S6 + reproducible script). Remaining items are ADDRESSED-IN-SUPPLEMENT, not open.

---

## Cross-reviewer summary

> **[v0.24]** The table below is the *original* snapshot. As of v0.24
> the "new session needed" column has been worked through: the planned
> Sessions 4–6 happened, and the distillation arc (47→19pp) plus the
> supplement resolved the rest. Current disposition: nearly all asks
> **CLOSED** or **ADDRESSED-IN-SUPPLEMENT**; a small set of continuous-
> action / minor items **DECLINED-WITH-RATIONALE** (out of discrete-core
> scope, preserved in `draft-monograph/`).

| Category | Already done | Quick fix this turn | New session needed |
|---|---|---|---|
| Reviewer 1 content asks | 4 of 12 | 0 | 8 |
| Reviewer 2 structural asks | 3 of 22 | 3 | 14 (mostly small) |

**Both reviewers agree on the core contribution** — the eight-axis
problem-first framework. Neither asks the team to abandon it. **[v0.24]**
The frame survived intact (W7 relabeled composite W7a/W7b; §IV.I/J
added as emerging directions); the distillation compressed pages
without abandoning the lens.

**Headline gap analysis:**

1. **Two content gaps** are individually large:
   - Q-learning for LLM/VLM alignment (Reviewer 1 #7) — natural new §IV.J or major §IV.G expansion
   - Predictable scaling laws (Reviewer 1 #6) — directly updates our §IV.G claims

2. **One scholarly apparatus gap** is non-negotiable for top-venue publication:
   - The bibliography. Already on our outstanding list. Needs pandoc-citeproc + `.bib` wiring before any submission, regardless of how much content we add.

3. **One methodological gap** would substantially strengthen the paper:
   - PRISMA-style review methodology in §III.A (search strings, screening flow, date cutoff, exclusion log). This is what makes a *narrative survey* defensible as a *systematic review*.

4. **One taxonomic gap** is cheap and worth doing:
   - Split or relabel W7 — Reviewer 2's cleanest taxonomic critique

5. **Two minor framing fixes** would inoculate against pushback:
   - One paragraph framing the distributional RL W4 placement as "an interpretive lens, defended below" rather than canonical (§IV.D.A)
   - One paragraph in §I pointing to the prior-art sweep when claiming "first multi-axis problem-first survey"

---

## Recommended next sessions

> **[v0.24] HISTORICAL — these sessions were executed.** Session 4
> apparatus (bibliography, §III PRISMA, limitations/ethics, rliable +
> Castro cites, W4/firstness framing) is CLOSED. Session 5 large
> content (§IV.J LLM/VLM alignment; §IV.G scaling reconciliation;
> PDQN) is CLOSED or supplement-routed. Session 6 refinement (W7
> composite relabel; §VII version pins → supp S4; §VI seeds/CIs +
> CVPI/MPI defs + planning-oracle framing) is CLOSED. Continuous-action
> items (Adjoint Matching, HJB) DECLINED-WITH-RATIONALE. Kept below for
> the record.

If the team has bandwidth for three more sessions before final
submission:

**Session 4 — quick wins + scholarly apparatus (highest priority).**
- ⚙ Fixed in this turn: PQN year (2024 not 2025), MDP notation θ removed, §VII.C count corrected
- ⊕ Stand up the References section: wire pandoc-citeproc, build `.bib` file, audit every in-text citation
- ⊘ Add PRISMA-style review protocol to §III.A
- ⊘ Add limitations/ethics subsection
- ⊘ Cite Agarwal et al. 2021 (rliable / Statistical Precipice) and Castro et al. 2020 (Revisiting Rainbow) in §V and §VI
- ⊘ One paragraph framing distributional RL W4 placement as interpretive (§IV.D.A); one paragraph pointing at prior-art sweep in §I

**Session 5 — large content gaps (most reviewer impact).**
- ⊘ §IV.J "Q-learning for Large Language Model and Vision-Language Model Alignment" — ShiQ, VLM Q-Learning, Q♯, KL-regularized Q-learning. Natural new ninth axis-section (the eight-axis frame already broken to nine by §IV.I theoretical advances)
- ⊘ §IV.G integration of Rybkin et al. predictable scaling laws — restructures the section's "single-learner Q is sample-bottlenecked" framing
- ⊘ §IV.E.B.5 addition: Q-learning with Adjoint Matching alongside FQL
- ⊘ §IV.A addition: Parameterized Q-Networks for hybrid action spaces

**Session 6 — taxonomic refinement and remaining methodological asks.**
- ⊘ Split or composite-label W7
- ⊘ §IV.D: crossing-quantiles problem + non-crossing variants
- ⊘ §VII: repository audit version pinning; named-vs-feature-support distinction
- ⊘ §VI: code/seed/hyperparameter release or claim softening; CVPI/MPI definitions
- ⊘ §VI baseline framing (planning baselines as oracles)

**Beyond three sessions:**
- Author block fill-in
- Two-column layout (deferred; needs IEEEtran or longtable workaround)
- Visual polish (TikZ genealogy, table overflow)

---

## Honest note

**Reviewer 1's report is significantly more positive than Reviewer 2's.** This may reflect different reviewer standards (Reviewer 1 reads as more enthusiastic, Reviewer 2 reads as more rigor-focused). For a top-tier venue, Reviewer 2's bar is the operative one — the bibliography gap and methodological transparency asks are the kind that block acceptance regardless of content quality. Closing Reviewer 1's content gaps without closing Reviewer 2's scholarly apparatus gaps would not be enough.

The path forward is therefore *both*: fix scholarly apparatus first
(Session 4), close the largest content gaps second (Session 5), then
refine (Session 6).

---

*Audit grounded in: two new reviewer reports (provided in chat),
draft files in `draft/`, prior audit `10-reviewer-audit.md`.*

*[v0.24 sync, 2026-05-30] Re-assessed against the distilled 19-page
TAI draft (`draft-tai/`) + 10-page supplement (S1–S6). The
scholarly-apparatus bar Reviewer 2 set is now met: bibliography wired,
§III PRISMA methodology, §VI 100 seeds + bootstrap CIs with planning-
oracle separation, auditable supplement + reproducible script.
NOTE: this second-round audit is SEPARATE from the three LLM reviews
of v0.14, which are triaged in `14-llm-review-triage.md` — do not
merge the two rounds.*

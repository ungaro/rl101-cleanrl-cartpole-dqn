# After Reading the Draft — Sharpened Direction

Companion to `01-pitch-analysis.md`. That note was written from the
pitch deck alone; this one revises the suggestions now that I've read
the full 20-page draft (`IEEE_AI_Journal_on_Q_Learning_New_Version.pdf`,
May 29 2026).

The core insight from `01` survives: **the paper is organized as a
catalogue, the reviewers want analysis.** But the draft is further along
than I assumed, and a few of my earlier suggestions need to be sharpened
or replaced.

---

## What changed after reading the draft

### 1. The "Critical Reflections" sections already exist — but they're shallow

Each Related Works subsection (A through F) ends with a "Critical
Reflections" paragraph. This is good — the bones of synthesis are
already there. The problem is the *content*:

- Most reflections read as *open question prompts* ("future work could
  explore X") rather than *substantive analysis* ("method Y works on
  axis A but degrades on axis B because of mechanism C").
- They live inside each method-type bucket, so they cannot make
  cross-bucket comparisons (e.g. "distributional methods solve the same
  variance problem that ensembles solve, but via a different mechanism
  — here's the trade-off").

**Implication:** the structural pivot in Suggestion 1 is still the
right move, but the framing changes — we're not *adding* synthesis
sections, we're *upgrading and cross-linking* the synthesis that's
already there.

### 2. The modern-RL gap is bigger than the deck suggested

The deck framed reviewer point 5 ("need more modern RL") as a relatively
small expansion. The draft makes clear the gap is structural:

| Modern Q-learning area | In the draft? |
|---|---|
| Offline RL (CQL, IQL, BCQ, EDAC) | **No** |
| Multi-agent (QMIX, VDN, QPLEX) | **No** |
| Distributed scale (Ape-X, R2D2, Agent57) | **No** |
| Meta-RL / continual Q-learning | **No** |
| Hindsight Experience Replay (HER) | **No** |
| Categorical post-2019 work beyond FQF | **No** |
| World-models with Q-targets (MuZero family) | **No** |

The newest non-2025 deep-RL entries are MeDQN (2023), PSDQN (2023), and
Rainbow (2018). The post-2018 deep-RL coverage is **two papers**. CBDQ
(2025) and PQN (2025) round out the modernity story, but neither
represents the directions the field actually moved.

**Implication:** "add a Modern RL appendix" is not enough. The taxonomy
itself doesn't have a slot where offline RL or multi-agent value
decomposition belongs. This is what makes Suggestion 1 (problem-first
reorganization) load-bearing rather than ornamental — those families
naturally land in *distribution shift* and *coordination* buckets that
the current taxonomy doesn't admit.

### 3. The repository comparison is much stronger than I feared

I worried in `01` that the repository section reads as a sidebar. Re-read:
it does — but Tables VII and VIII are genuinely useful artifacts. The
Table VII coverage matrix is exactly the kind of concrete synthesis the
reviewers want; it just isn't *linked* to the method discussion.

**Implication:** my earlier suggestion to "fold each repo's choices into
the method sections they implement" is the right move, but it should be
lightweight — a one-line callout in each method's writeup ("supported
in Tianshou, XuanCe; absent from CleanRL, SB3 — see §VII") rather than
a structural rewrite.

### 4. The Atari benchmark tables are the strongest argument for the structural pivot

Tables II and III are full of dashes — that's the reviewer's "Atari is
limited" complaint made visceral. The current framing apologizes for the
gaps ("Reporting gaps limit fair ranking"). The problem-first
reorganization *uses* the gaps as evidence: if FQF dominates dense-reward
games but Montezuma's Revenge sits at 0 across the entire table, that's
not a reporting gap — that's *the central limitation of value-based
exploration*, and it belongs in the "Brittle exploration" section as
empirical motivation.

**Implication:** the Atari tables don't need *more data* (we already
flagged that they cannot be expanded faithfully). They need to be
*re-purposed* as evidence in a problem-first narrative.

### 5. The Conclusion already promises a Q-learning–specific repository

"Looking forward, a clear avenue for future research lies in the
development of a dedicated, community-maintained repository focused
exclusively on Q-learning and its deep variants."

This is a real spin-off opportunity and a way to give reviewers
something concrete: even if the repo is just a coverage-tracker plus a
stub for the missing variants (Bootstrapped DQN, EBQL, UCB Ensemble,
PSDQN, MeDQN, CBDQ — *none* of which are in any of the six surveyed
repos per Table VII), the act of building it converts the survey's
findings into a community deliverable.

**Implication:** this could be the "single concrete commitment" the
revision needs. A barebones GitHub repo with the table-VII matrix as
its README and a roadmap is achievable in a week.

---

## Revised suggestion set

`01` proposed two suggestions. After reading the draft, I'd revise to
**three**, ordered by leverage:

### A. (highest leverage, structural) Problem-first reorganization

Same as Suggestion 1 in `01`, but sharpened by the modern-RL gap finding
above. The pitch to the team:

> The taxonomy has six method-type buckets. The reviewers' five
> complaints all stem from this organization. The modern-RL methods we
> are missing (offline, multi-agent, distributed) cannot be added to
> the current taxonomy without an "and also..." appendix. Reorganizing
> around *which weakness of vanilla Q the method addresses* lets us
> (i) make the cross-method comparisons reviewers want, (ii) re-purpose
> the Atari tables as evidence rather than apologize for their gaps,
> (iii) add modern families into named, principled slots.

**Concrete first-week deliverable:** a section-by-section re-mapping
document. For each of the 30-odd methods in the current draft, name the
weakness it primarily addresses and which axis-section it would live in.
A few methods (Rainbow, distributional, ensembles) appear in multiple —
that's the point.

### B. (medium leverage, visual) Centerpiece genealogy figure

Same as Suggestion 2a in `01`. After reading the draft, I'd argue this
becomes the **figure that captures suggestion A's claim visually**:
nodes for methods, edges annotated with the weakness the child
addresses in the parent.

This is also the cheapest way to make modern-RL absence visible. Once
the genealogy is drawn, CQL/IQL/QMIX appear as missing branches in
specific places — which is itself an argument for the structural
pivot.

**Concrete first-week deliverable:** ASCII or TikZ first pass. The
discussion of "which edge label goes where" *is* the seed of the
section-by-section remap in A.

### C. (low leverage but high signal-to-effort) Spin off the Q-learning repo

Take the Conclusion's promise seriously and stand up a stub repo. README
is Table VII. Each missing-from-everywhere entry (Bootstrapped DQN, EBQL,
UCB Ensemble, PSDQN, MeDQN, CBDQ) gets a tracking issue with the source
paper, the algorithm sketch, and a "wanted: implementation" tag.

This costs ~1 day. It is a concrete deliverable reviewers can point at:
"the survey produced a community artifact." It doesn't require any
prose changes to the paper.

**Concrete first-week deliverable:** GitHub repo skeleton, README,
issues for the six absent variants.

---

## Strategic recommendation (revised)

Run **B + C this week**, in parallel; pitch **A as the next phase**.

- **B (genealogy figure)** is the lowest-risk synthesis artifact and is
  worth a week of one person's time on its own merits.
- **C (repo spin-off)** is ~1 day of work and converts a Conclusion
  bullet into a concrete deliverable. It also keeps momentum visible
  to reviewers.
- **A (structural pivot)** is the right intellectual move but is a 2–4
  week multi-author exercise. Pitch it as the work for the next
  revision cycle, with the genealogy figure as the framing artifact
  that the structural pivot will operationalize.

If the team has only enough bandwidth for one of these, do **B** — the
figure is the single artifact that most directly addresses reviewer
points 1 (algorithms feel isolated) and 2 (need conceptual insight),
and it can be inserted into the draft with minimal prose disruption.

---

## What I'd want from the team before committing to a direction

1. **How locked is the taxonomy?** If the six categories are
   load-bearing in the cover letter or already accepted by the editor
   as the paper's contribution, the structural pivot becomes more
   expensive than I think.
2. **Is there appetite for a co-author taking ownership of modern
   families?** Offline RL alone (CQL/IQL/BCQ/EDAC) is a real lift; a
   PhD researcher with offline-RL chops would be the right owner.
3. **What's the editor's tone?** "Major revision" vs. "reject and
   resubmit" maps to different ambition levels.

---

## Open question I cannot answer from inside the draft

Are there other 2024–2026 Q-learning surveys we'd be competing against?
A 30-minute Google Scholar + arXiv sweep before committing direction
would be cheap insurance. If someone else is already doing the
problem-first organization, our angle changes — we lean into the
repository comparison + tabular benchmarking, which is genuinely
unique. If no one is, the structural pivot is also a differentiation
claim.

---

*Drafted on branch `alp/q-survey-revision-prep`. Not merged to main.*

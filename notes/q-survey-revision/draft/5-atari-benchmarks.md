# Section V — Atari Benchmark Analysis (revised draft)

Reframes the prose around Tables II and III to use task-category
stratification as evidence for the axis-stratified claim. Replaces
the current apologetic framing of reporting gaps with an analytical
one in which the gaps themselves carry information.

Tables II and III are preserved structurally; the surrounding prose
is rewritten. ~800 words.

The optional grouped-bar figure proposed in `08-figure-proposals.md`
(F-B2) would sit naturally inside this section as a visual complement
to Tables II/III.

---

## V. ATARI BENCHMARK ANALYSIS

This section synthesizes Atari benchmark performance reported across
the methods of §IV. Because third-party implementations often diverge
from authors' codebases, and because official repositories are
incomplete (§VII), we extract results directly from peer-reviewed
publications rather than re-running experiments. Tables II and III
present raw per-game scores under each paper's original evaluation
protocol; Figure F-B2 (if included) provides an axis-stratified
visual summary.

### A. Table structure and dual-view organization

Tables II and III are structured on two axes simultaneously, and
both groupings are load-bearing.

**Row grouping (the legacy six-category taxonomy).** Methods are
grouped into six rows: *Statistical Methods*, *Q-Function
Computation*, *Memory/Replay*, *Ensemble-Based*, *Model-Based*, and
*Pure Q-Learning*. This grouping is preserved from the original
draft and matches the method-type taxonomy of prior Q-learning
surveys [12]–[15]. The revised paper retains this row organization
in Tables II/III as a presentation device, because (a) it lets
readers familiar with the conventional taxonomy navigate the
empirical data without translation; (b) it preserves the
within-family comparability that the original draft worked to
establish; and (c) the six-category grouping makes the modern-RL
absences (§IV.E offline, §IV.F multi-agent, §IV.G distributed)
visually salient — entire columns of the matrix that the legacy
taxonomy has no row to hold.

**Column grouping (task categories).** Atari games are grouped into
seven task categories — Reaction-Time Control, Strategic Planning,
Sparse Rewards, Dense Rewards, Large Observation Space, Partially
Observable Environments, Stochastic Environments. This grouping is
also preserved from the original draft.

**Axis annotation (new).** A new rightmost column or footnote
appended to each table maps each method's row to its primary §IV
axis. The annotation does not change the table's structure; it
provides a third lens. Readers can read Table II as
"method-family × task-category" (the original draft's view) or as
"primary-axis × task-category" (the revised view) by attending to
either the row grouping or the new axis column.

This dual-view organization is itself a contribution. Reviewers
arriving with the catalogue-style expectation see the familiar
six-category structure; reviewers seeking analytical synthesis read
the prose subsection-by-subsection below and the new axis column.
Neither audience needs to translate the empirical data to align
with their reading.

### B. Task category stratification

The current draft's Tables II and III group Atari games into seven
categories — Reaction-Time Control, Strategic Planning, Sparse
Rewards, Dense Rewards, Large Observation Space, Partially Observable
Environments, and Stochastic Environments. These categories were
introduced to provide a structured view of where methods excel; in
the revised paper they take on additional load-bearing significance.
**Each task category preferentially tests one or two of the eight
weaknesses introduced in §II.B**:

| Task category | Diagnoses weakness(es) |
|---|---|
| Reaction-Time Control | function-approx stability (W8), sample inefficiency (W2) |
| Strategic Planning | reward sparsity & credit assignment (W4) |
| Sparse Rewards | brittle exploration (W3) |
| Dense Rewards | none specifically — baseline for stability |
| Large Observation Space | function-approx stability (W8) |
| Partially Observable | slow adaptation / recurrence (W7) |
| Stochastic Environments | overestimation bias (W1), distribution robustness |

The mapping is not perfect — most games test multiple weaknesses to
some degree — but the dominant signal in each category aligns
cleanly with one or two axes. The empirical claim of this paper's
structural pivot is that methods targeting weakness $W_i$ should
excel on the diagnostic category for $W_i$ and remain at baseline
elsewhere. Tables II and III, read through this lens, support the
claim.

### C. Patterns by axis

**Brittle exploration (W3, §IV.C).** The sparse-reward category
(Montezuma's Revenge, Pitfall!, Private Eye) is where the axis
matters most. Methods targeting exploration explicitly — NoisyNet,
Bootstrapped DQN, CBDQ, Posterior Sampling DQN — produce nonzero
scores on Montezuma where vanilla DQN and methods targeting other
axes do not. Rainbow's 384 on Montezuma is attributable specifically
to its NoisyNet component (per the original ablation). DQfD's 4,638
on Montezuma — substantially higher than any non-demonstration
method — illustrates the trade-off explored in §IV.B: demonstration
substitutes for directed exploration. RND and Go-Explore (newly
discussed in §IV.C but not in Tables II/III) report Montezuma scores
of ~10,000 and >1 million respectively, indicating that the axis is
not saturated by the methods covered here.

**Reward sparsity and credit assignment (W4, §IV.D).** The
strategic-planning category (Q*bert, Ms. Pac-Man) is where
distributional methods produce their most striking results. QR-DQN's
572,510 on Q*bert is the single largest score in our compilation by
any method — over $25\times$ the best non-distributional result on the
same game — and the pattern is consistent across the category. IQN
and FQF follow the same pattern at lower absolute magnitude.

**Function-approximation stability (W8, §IV.H).** Reaction-Time
Control games (Breakout, Space Invaders, Enduro) show comparatively
small inter-method variance: Double DQN, Dueling DQN, Rainbow, and
distributional methods cluster within a factor of two of each other
on these games. The pattern indicates that for games where the
weakness signal is *stability rather than capability*, modest
algorithmic improvements over the DQN baseline suffice — and PQN's
strong performance on the category (§IV.H) is consistent with this
interpretation.

**Sample inefficiency (W2, §IV.B).** PER's performance pattern,
across the categories, is most pronounced on strategic-planning and
sparse-reward games — categories where high-information transitions
are rare. On dense-reward games where informative transitions are
abundant under uniform sampling, the PER advantage is small. The
ablation pattern from Rainbow (PER is the largest single contributor)
is consistent with this category-stratified picture.

### D. The reporting gaps as evidence

Dashes ("---") in Tables II and III indicate that the original paper
did not report a score for that game under the stated evaluation
protocol. The current draft notes that "absence of data does not
imply poor performance"; this is true but understates the
information content of the gaps.

**For most methods, the gap pattern is itself diagnostic.** A method
that reports comprehensively on dense-reward games and skips sparse-
reward games signals — perhaps implicitly — that the method does not
solve the sparse-reward problem. A method that reports on Reaction-
Time Control and skips Stochastic Environments is unlikely to be a
strong response to W1 (overestimation, which compounds under
stochasticity). The pattern of dashes, considered as a structured
absence, is sometimes more informative than the present scores.

Several specific gaps illustrate the point. Ensemble Bootstrapping
[46] reports on 11 of 57 games and skips the entire sparse-reward
category, despite the ensemble mechanism being a natural exploration
candidate. Memory-Efficient DQN [41] reports on 5 of 57 games
selected for memory-sensitivity, leaving the method's broader
performance profile undefined. The dashes are honest about reporting
scope; reading them as evidence is internally consistent with the
authors' own treatment of their methods.

### E. The "improvement ladder" and its limits

Within several axes — particularly W1 (overestimation) and W4
(credit assignment) — a consistent chronological improvement
trajectory is visible: classical DQN → Double DQN → Dueling /
Prioritized → Distributional / Quantile-based methods, with each
generation outperforming the previous on its diagnostic category.
This "ladder" is real but axis-specific. A method that places at
the top of the W4 ladder (FQF) does not necessarily improve on W3
(FQF scores zero on Montezuma). The ladder is a strong frame *within
an axis* and a weak frame *across axes* — which is the central
empirical argument for the structural pivot.

### F. Why we do not produce a leaderboard

Earlier surveys [13]–[15] bold or italicize the "best" result per
game. This presentation implicitly invites comparison across methods
that used different evaluation protocols (training-frame budgets,
seed counts, no-op-start variations, sticky-action settings).
Boldface in the presence of protocol divergence overstates the
strength of comparisons. We refrain from highlighting per-cell
best-results in Tables II/III; the axis-stratified narrative above
substitutes evaluation against the *diagnostic categories* for the
weaker comparison against per-game best.

---

*Notes for integration:*
- Tables II and III are preserved structurally; only the surrounding
  prose changes. The current "Several caveats and trends" bulleted
  list is replaced by the §V.A-E structure above.
- If figure F-B2 (`08-figure-proposals.md`) is added, it sits
  naturally between §V.A (the category-to-axis mapping table) and
  §V.B (the patterns subsection). The figure visualizes the
  category-stratification claim quantitatively.
- The "improvement ladder" framing in §V.D is a sharpening of the
  current draft's "Improvement ladder" bullet but explicitly notes
  the ladder's axis-locality — the central caveat that distinguishes
  the new framing from the old.

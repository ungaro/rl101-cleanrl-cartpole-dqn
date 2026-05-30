# III. Methodology {#sec-iii}

This review surveys Q-learning algorithms spanning over three decades
of development, from foundational tabular approaches to recent deep
variants. We include influential papers that shaped the theoretical
landscape, introduced practical improvements, or demonstrated wide
applicability in real-world domains. The selection prioritizes
methods whose contributions can be mapped onto one or more of the
eight weakness axes introduced in §II.B — methods that respond to a
recognizable weakness of vanilla Q-learning with a distinct
mechanism.

### A. Source selection and review protocol

This paper is best characterized as a *narrative review with
empirical add-ons* rather than a strictly systematic review in the
PRISMA sense: the synthesis is interpretive (each axis carries an
argued reading of the methods it surveys), and the eight-axis
framework was developed iteratively with the literature rather than
applied as a pre-registered classification. We document the search
and selection protocol here so that the resulting axis assignments,
benchmark extractions, and repository comparisons can be
independently audited.

**Databases and search strategy.** Source candidates were
identified from three databases: Google Scholar (broad coverage,
citation-graph traversal), arXiv (cs.LG and cs.AI listings, primary
source for preprints), and Semantic Scholar (citation-relationship
analysis). The search ran from December 2025 through May 2026 with
a cutoff date of 2026-05-15 for the late-2025 / 2026 paper sweep
documented separately. Searches combined topical anchors
("Q-learning", "deep Q-network", "DQN") with axis-specific terms
("overestimation bias", "prioritized replay", "exploration bonus",
"distributional reinforcement learning", "offline reinforcement
learning", "value decomposition", "distributed reinforcement
learning", "meta-reinforcement learning", "function approximation
stability") and recency filters (publication date 2018-onward for
deep-RL methods; full historical range for foundational results).

**Inclusion and exclusion criteria.** Five inclusion criteria were
applied:

1. **Method centrality** — the work introduces a methodological
   contribution to the Q-learning family rather than applying
   Q-learning to a domain.
2. **Mechanism distinctness** — the contribution is mechanistically
   distinguishable from prior work along at least one of the eight
   axes of §II.B.
3. **Empirical validation** — the work reports results on a
   recognized benchmark (Atari, classic control, D4RL, SMAC,
   ProcGen, or equivalent).
4. **Cross-reference impact** — the work is cited as foundational
   by subsequent methods in the same axis-family, *or* the work was
   published within the previous twelve months and represents an
   active research thread.
5. **Reproducibility** — code is publicly available or the
   algorithm is sufficiently specified to be independently
   re-implemented.

Exclusion criteria removed: (a) application-domain papers using
Q-learning as a black-box tool without methodological contribution;
(b) workshop-only or non-archival preprints lacking subsequent
follow-up; (c) methods that have been demonstrably superseded by
equivalent-effort alternatives (e.g., we cover Maximin Q-Learning
but not all of its precursors). Methods satisfying all five
inclusion criteria received per-paper treatment. Methods satisfying
a subset received compact treatment as part of a family (e.g., the
value-decomposition family in §IV.F is treated through its
canonical members VDN, QMIX, QPLEX, QTRAN, with QFIX added on the
basis of criterion 4's twelve-month clause).

**Screening flow.** Initial candidate identification produced
approximately 200 papers; first-pass title/abstract screening
narrowed this to approximately 120. A subsequent full-text
screening against the five inclusion criteria produced
approximately 80 papers receiving full per-method treatment, with
an additional 20 cited at family or cross-reference level. We do
not publish a strict PRISMA flow diagram here because the
search-and-iterate process was non-linear: as the eight-axis
framework crystallized, several methods that initially appeared
incidental were promoted to per-paper status, and conversely some
once-canonical entries were demoted to family-level mention. We
treat this iteration as a methodological *limitation* rather than a
defect (and surface it explicitly in §VIII.D); a strictly
systematic review would have required pre-registering the axis
framework before the literature search, which we did not.

**Axis-assignment protocol.** Each method covered in §IV is
assigned to a *primary axis* — the weakness whose response motivated
the method's introduction — and zero or more *secondary axes* —
additional weaknesses the method incidentally addresses or
partially mitigates. Primary assignment is made on the basis of the
method's stated motivation in its original publication and the
predominant mechanism of its contribution. Where the original
publication's motivation differs from our axis assignment (notably
for distributional methods in §IV.D and Dueling DQN in §IV.H), the
re-interpretation is argued explicitly within the section. The
complete per-method mapping, including secondary axes and
cross-references, appears in Appendix A.

**Per-paper extraction template.** For methods receiving per-paper
treatment, we extracted: (i) the formal weakness statement the
method addresses (§IV.X.A reference); (ii) the key mechanism-
defining equation; (iii) the primary empirical evidence reported in
the original publication; (iv) trade-offs explicitly discussed by
the authors; (v) any subsequent ablations or follow-up evaluations
relevant to the trade-off discussion. This extraction template is
not published as supplementary material in the current version, but
the per-method writeups in §IV reflect its structure consistently.

### B. Comparison with prior surveys

To position this survey against existing work, we compiled Table I,
which compares this paper with five prior Q-learning–focused surveys
[12]–[15] and [Ghasemi 2024/2025] along five distinguishing
dimensions: analysis of public DQN repositories, unified taxonomy
spanning tabular and deep Q-learning, extraction of original-paper
Atari benchmarks, classic control benchmarks from controlled
re-implementation, and per-paper review organized around mechanism
and axis. Where prior surveys focus on chronological or method-type
organization, this paper's problem-first axis structure (§IV) is, to
our knowledge, the first multi-axis problem-first treatment of
Q-learning specifically. This claim is supported by a documented
2024-2026 sweep across Google Scholar, arXiv (cs.LG and cs.AI), and
Semantic Scholar that surveyed every recent Q-learning, DQN, and
broad-RL survey we could identify; the closest prior art is the
single-axis problem-first organization of [Springer NCAA 2026],
which addresses distribution shift in offline RL only. Search
strings, dates, and per-survey notes are recorded as supplementary
material so the claim can be independently audited.

\begin{table*}[t]
\centering
\caption{Comparison of Q-Learning and Deep Q-Learning Survey Papers. \\$\bullet$ = discussed; $\circ$ = not discussed; partial = \textit{partial}.}
\label{tab:survey-comparison}
\small
\begin{tabular}{p{0.42\linewidth}cccccc}
\toprule
Aspect & Urtans 2018 & Jang 2019 & Boppiniti 2021 & Hafiz 2022 & Ghasemi 2024/25 & Ours (2026) \\
       & [12]        & [13]      & [14]           & [15]       &                 &              \\
\midrule
Analyzes Public DQN Code Repositories & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\bullet$ \\
Unified Taxonomy Covering Both Tabular Q and DQN & $\circ$ & $\circ$ & $\circ$ & $\circ$ & \textit{partial} & $\bullet$ \\
Atari Benchmark Results Extracted from Prior DQN Papers & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\bullet$ \\
Classic Control Benchmark Results from Original Implementations & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\bullet$ \\
Thorough Per-Paper Literature Review & $\circ$ & $\circ$ & $\circ$ & $\circ$ & \textit{partial} & $\bullet$ \\
Problem-First Organization Across Multiple Axes & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\circ$ & $\bullet$ \\
\bottomrule
\end{tabular}
\end{table*}

### C. Axis assignment methodology

Each method covered in this paper is assigned to a *primary axis* —
the weakness whose response motivated the method's introduction — and
zero or more *secondary axes* — additional weaknesses the method
incidentally addresses or partially mitigates. Primary assignment is
made on the basis of the method's stated motivation in its original
publication and the predominant mechanism of its contribution.
Where the original publication's motivation differs from our axis
assignment (notably for distributional methods in §IV.D and Dueling
DQN in §IV.H), the reinterpretation is argued explicitly within the
section.

Cross-references between axis-sections — e.g., Rainbow appearing in
§IV.B (primary) and being cross-referenced from §IV.A, §IV.C, §IV.D,
and §IV.H — surface the connective tissue that catalogue-style
surveys obscure. Appendix A provides a legacy indexer mapping each
method to both the problem-first axis assignment and the prior
method-type taxonomy, supporting readers who arrive expecting the
older organization.

### D. Empirical evidence sources

Section V analyzes Atari benchmark results extracted from the
original papers introducing each method. Section VI reports tabular
benchmarks from controlled re-implementations on Gymnasium
environments (FrozenLake-v1, Taxi-v3, CliffWalking-v1). Section VII
analyzes algorithmic coverage across six widely-used open-source
deep RL repositories (Tianshou [6], XuanCe [7], CleanRL [8], DQN
Zoo [9], Stable Baselines3 [10], RLlib [11]). The three evidence
streams together support different aspects of the paper's
contribution: the literature analysis demonstrates methodological
diversity, the controlled experiments isolate algorithmic from
architectural effects, and the repository analysis maps the
practical landscape of available implementations.

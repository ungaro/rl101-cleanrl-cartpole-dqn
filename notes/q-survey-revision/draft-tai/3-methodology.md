# III. Methodology {#sec-iii}

This section states our search, selection, and classification protocol
so that the resulting axis assignments, benchmark extractions, and
repository comparisons can be independently audited. We adopt a
systematic, PRISMA-style protocol: an explicit search across named
databases, fixed search strings, a stated date range, and recorded
inclusion/exclusion criteria. The synthesis remains interpretive — each
axis carries an argued reading of the methods it surveys — and the
eight-axis framework of [§II.B](#sec-ii) was refined against the
literature, which we treat as a limitation surfaced in
[§VIII.D](#sec-viii) rather than a defect. The full search log is
provided as supplementary material.

**Search protocol.** Candidates were drawn from three databases:
Google Scholar (broad coverage, citation-graph traversal), arXiv
(cs.LG and cs.AI, primary preprint source), and Semantic Scholar
(citation-relationship analysis). The recent-work sweep ran from
December 2025 through May 2026 (cutoff 2026-05-15), augmented by a
full historical pass for foundational results. Searches combined
topical anchors ("Q-learning", "deep Q-network", "DQN") with
axis-specific terms ("overestimation bias", "prioritized replay",
"exploration bonus", "distributional reinforcement learning",
"offline reinforcement learning", "value decomposition", "distributed
reinforcement learning", "meta-reinforcement learning", "function
approximation stability"), with a 2018-onward recency filter for
deep-RL methods and no date floor for foundational results.

**Inclusion and exclusion criteria.** A work was included if it met
all five criteria: (1) *method centrality* — it contributes to the
Q-learning family rather than applying Q-learning to a domain;
(2) *mechanism distinctness* — it is mechanistically distinguishable
from prior work along at least one [§II.B](#sec-ii) axis;
(3) *empirical validation* — it reports results on a recognized
benchmark (Atari, classic control, D4RL, SMAC, ProcGen, or
equivalent); (4) *cross-reference impact* — it is cited as
foundational by later methods in its axis-family, *or* appeared within
the prior twelve months as an active thread; and (5) *reproducibility*
— code is public or the algorithm is fully specified. We excluded
(a) application papers using Q-learning as a black box, (b)
workshop-only or non-archival preprints without follow-up, and
(c) methods superseded by equivalent-effort alternatives. Works
meeting all five received per-paper treatment; others were folded into
family-level coverage (e.g., the value-decomposition family in
[§IV.F](#sec-iv-f) via VDN, QMIX, QPLEX, QTRAN, with QFIX added under
criterion 4's twelve-month clause).

**Screening flow.** Initial identification produced ~200 papers;
title/abstract screening narrowed this to ~120; full-text screening
against the five criteria yielded ~80 receiving per-method treatment,
plus ~20 cited at family or cross-reference level. The complete
ledger is supplementary material.

**Comparison with prior surveys.** Table I positions this paper
against five prior Q-learning–focused surveys
[@urtans_2018_pygame; @jang_2019_qsurvey; @boppiniti_2021_evolution; @hafiz_2023_dqnsurvey]
and [@ghasemi_2024_rlsurvey] along six distinguishing dimensions.
Where those surveys organize chronologically or by method type, this
paper's problem-first axis structure ([§IV](#sec-iv)) is, to our
knowledge, the first multi-axis problem-first treatment of Q-learning;
the closest prior art, [@springer_2026_distshift], is single-axis
(distribution shift in offline RL only). Per-survey notes are recorded
as supplementary material.

\begin{table*}[t]
\centering
\caption{Comparison of Q-Learning and Deep Q-Learning Survey Papers. \\$\bullet$ = discussed; $\circ$ = not discussed; partial = \textit{partial}.}
\label{tab:survey-comparison}
\small
\begin{tabular}{p{0.42\linewidth}cccccc}
\toprule
Aspect & Urtans 2018 & Jang 2019 & Boppiniti 2021 & Hafiz 2022 & Ghasemi 2024/25 & Ours (2026) \\
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

**Axis-assignment methodology.** Each method covered in
[§IV](#sec-iv) is assigned a *primary axis* — the weakness whose
response motivated its introduction, determined from the stated
motivation in the original publication and the predominant mechanism
of its contribution — and zero or more *secondary axes* for weaknesses
it incidentally addresses. Where our assignment departs from the
original motivation (notably distributional methods under credit
assignment in [§IV.D](#sec-iv-d) and Dueling DQN in
[§IV.H](#sec-iv-h)), the reinterpretation is argued in-section.
Cross-references — e.g., Rainbow's primary placement in
[§IV.B](#sec-iv-b) with links from [§IV.A](#sec-iv-a),
[§IV.C](#sec-iv-c), [§IV.D](#sec-iv-d), and [§IV.H](#sec-iv-h) —
surface connective tissue that catalogue-style surveys obscure. The
complete per-method mapping, with secondary axes and a full index for
readers expecting the older organization, is supplementary material.

**Empirical evidence sources.** [§V](#sec-v) analyzes Atari results
extracted from the original papers; [§VI](#sec-vi) reports tabular
benchmarks from controlled re-implementations on Gymnasium
(FrozenLake-v1, Taxi-v3, CliffWalking-v1); and [§VII](#sec-vii)
analyzes algorithmic coverage across six open-source deep-RL
repositories (Tianshou [@weng_2022_tianshou], XuanCe
[@liu_2023_xuance], CleanRL [@huang_2022_cleanrl], DQN Zoo
[@quan_2020_dqnzoo], Stable Baselines3 [@raffin_2021_sb3], RLlib
[@liang_2018_rllib]). Together these streams demonstrate
methodological diversity, isolate algorithmic from architectural
effects, and map the practical implementation landscape.

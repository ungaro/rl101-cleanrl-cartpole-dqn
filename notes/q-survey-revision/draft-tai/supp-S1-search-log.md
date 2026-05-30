# S1. Systematic Search Log and Prior-Art Sweep {#supp-s1}

This appendix documents the systematic literature search supporting the
methodology of §III. It records the databases queried, the date range,
the search strings, the inclusion and exclusion outcome, and a structured
comparison against recent reinforcement-learning and Q-learning surveys.
It is provided so that the screening flow and the survey-positioning
claims in §III can be independently audited.

## S1.1 Databases and Date Range

Three databases were queried. Google Scholar provided broad coverage and
citation-graph traversal; arXiv (categories cs.LG and cs.AI) served as
the primary preprint source; and Semantic Scholar supported
citation-relationship analysis, including forward and backward citation
expansion seeded from the closest-matching surveys.

The recent-work sweep covered December 2025 through May 2026, with a
cutoff of 2026-05-15. A recency filter of 2018 onward was applied to
deep-RL methods; foundational tabular and theoretical results were
admitted with no date floor.

## S1.2 Search Strings

Queries combined topical anchors with axis-specific terms. The topical
anchors were "Q-learning", "deep Q-network", and "DQN". These were
conjoined with axis-specific terms including "overestimation bias",
"prioritized replay", "exploration bonus", "distributional reinforcement
learning", "offline reinforcement learning", "value decomposition",
"distributed reinforcement learning", "meta-reinforcement learning", and
"function approximation stability". A complementary set of queries
targeted recent survey literature directly, combining "reinforcement
learning" and "Q-learning" with "survey", "overview", and "taxonomy" to
identify works whose scope could overlap that of this paper.

## S1.3 Inclusion and Exclusion Outcome

Candidates were screened against the five inclusion criteria stated in
§III: method centrality, mechanism distinctness, empirical validation,
cross-reference impact, and reproducibility. Initial identification
produced approximately 200 records; title and abstract screening narrowed
this to approximately 120; full-text screening against the five criteria
yielded approximately 80 works receiving per-method treatment, plus
approximately 20 cited at family or cross-reference level. Excluded
records comprised application papers using Q-learning as a black box,
workshop-only or non-archival preprints without follow-up, and methods
superseded by equivalent-effort alternatives.

A dedicated survey-comparison sweep was run over the December 2025–May
2026 window to confirm the positioning claims of §III. No survey
published in the 2024–2026 window adopts a multi-axis problem-first
organization of the Q-learning family unifying tabular and deep methods.
The nearest organizational precedent applies a problem-first structure
along a single axis (distribution shift) and is restricted to offline RL.
The works identified by this sweep are summarized below.

## S1.4 Prior Surveys Identified

Four works were identified as overlapping or adjacent in scope. Three are
surveys; one is an empirical implementation study rather than a survey
and is noted as such. The following table summarizes each work's scope,
its overlap with the present survey, and the points of differentiation.

Work | Scope | Overlap with this survey | Differentiation
--- | --- | --- | ---
Ghasemi, Moosavi and Ebrahimi 2024/2025 [@ghasemi_2024_rlsurvey] | Broad RL across tabular and deep methods, organized by method family (value-based, policy, actor-critic), with practical challenges as a secondary lens | Covers core Q-learning methods (Q-learning, Double Q-learning, DQN, DDQN, Dueling DQN) and unifies tabular and deep treatment | Organized by method family rather than problem axis; finer method-type categories (prioritized replay, distributional, Rainbow) and the repository and Atari analyses of this survey are not present
Springer NCAA 2026 distribution-shift survey [@springer_2026_distshift] | Offline RL only, organized problem-first along a single axis (distribution shift), with four method buckets (Q-value restriction, uncertainty-based Q-restriction, policy constraint, uncertainty-based policy constraint) | Shares a problem-first organizing principle and addresses the distribution-shift axis treated in §IV.E | Single-axis and restricted to offline RL; this survey generalizes the problem-first structure to multiple axes spanning tabular and deep Q-learning
Murphy 2024/2025 [@murphy_2024_rloverview] | Textbook-style overview covering value-based, policy, model-based, multi-agent, and LLM-plus-RL methods, organized by method type | General coverage of value-based methods, including an explicit LLM-RL angle that intersects §IV.J | Organized by method type; does not provide a Q-learning-specific DQN taxonomy, repository comparison, or extracted Atari benchmarks
Hundal, Xiao, Cao, Dong and Rigger 2025 [@hundal_2025_interchangeable] | Empirical benchmark of PPO across Stable-Baselines3, CleanRL, Baselines, RLlib, and Tianshou over 56 Atari games (implementation audit, not a survey) | Studies the same open-source repository landscape examined in the repository analysis of §VII | An empirical reproducibility audit of one algorithm (PPO); this survey's repository treatment is taxonomic, mapping algorithmic coverage across repositories to identify Q-learning implementation gaps. The two approaches are complementary

Additional works surfaced by the sweep were assessed and excluded as
non-competing on scope grounds: a taxonomy of RL for robotics and control
(continuous control, minimal DQN coverage), a survey of explainable deep
RL (organized by explanation level), a study of Q-learning for
metaheuristic optimization (Q-learning as a controller rather than core
theory), and several application-domain surveys (network intrusion
detection, real-world robotics successes, reward models). None adopts a
problem-first Q-learning taxonomy and none overlaps the contributions of
this survey.

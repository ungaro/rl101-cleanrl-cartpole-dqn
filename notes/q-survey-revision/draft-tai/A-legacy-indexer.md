# Appendix A. Legacy Indexer: Six Categories vs. Eight Axes {#sec-app-a}

To support readers approaching the field through the conventional
method-type taxonomy used in prior Q-learning surveys
[@urtans_2018_pygame; @jang_2019_qsurvey; @boppiniti_2021_evolution; @hafiz_2023_dqnsurvey],
we retain that six-category scheme as a secondary index. The two views
are complementary: a reader interested in *all distributional methods*
navigates by method type, while a reader interested in *all methods
addressing brittle exploration* navigates by axis ([§IV.C](#sec-iv-c)).
The complete per-method mapping (50 methods, including the 18 that have
no home in the legacy six categories — offline RL, multi-agent, and
distributed families that postdate it) is provided as supplementary
material. The inverted view below is the analytically interesting one.

| Legacy category ($n$) | Where its methods land in §IV |
|---|---|
| Statistical (6) | §IV.C (2: noise-based); §IV.D (4: distributional) |
| Q-Function Comp. (8) | §IV.A (2); §IV.H (2); §IV.B/C/G one each; §V (1) |
| Memory/Replay (4) | §IV.B (all 4) — coherent |
| Ensemble-Based (3) | §IV.A (1); §IV.C (2) |
| Model-Based (4) | §IV.C (1); §V foundations (3) |
| Pure Q-Learning (5) | §V foundations (4); §IV.H (1) |

: Reverse view — how each legacy method-type category distributes
across the eight problem axes.

Three patterns make the case for the problem-first organization.
First, *Memory/Replay* is the only legacy category that stays
coherent — all four methods address sample inefficiency
([§IV.B](#sec-iv-b)) — so there the two taxonomies coincide. Second,
*Ensemble-Based* methods fragment: the same architectural mechanism
serves overestimation control ([§IV.A](#sec-iv-a)) and exploration
([§IV.C](#sec-iv-c)) through different mechanisms-of-use that the axis
view separates and the legacy view conflates. Third, *Statistical
Methods* split between noise-based exploration ([§IV.C](#sec-iv-c)) and
distributional credit assignment ([§IV.D](#sec-iv-d)) — the sharpest
demonstration that the conventional taxonomy groups by *what a method
is* rather than *what weakness it addresses*.

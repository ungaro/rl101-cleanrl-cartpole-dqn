# Section I — Introduction (revised draft)

Replaces the current Introduction. The pivot from "we built a unified
taxonomy" to "we reorganize Q-learning around the *problems* its
variants were designed to solve" is the key framing change. The five
distinguishing contributions are preserved but reframed.

Drafted for the IEEE TAI template — paragraph density matches the
current draft. ~700 words. Citations use the existing reference
numbers where possible; new entries (Ghasemi 2024/2025) need to be
added to the bibliography.

---

## I. INTRODUCTION

Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) have
emerged as powerful paradigms for solving complex sequential
decision-making problems, with applications ranging from Atari games
[1], [2] and robotic control [3] to autonomous driving [4]. Among
DRL approaches, two major families dominate: value-based methods,
which estimate action-value functions, and policy-based methods, which
directly optimize policy distributions.

While policy-based algorithms have shown strong performance in tasks
with continuous action spaces, they often suffer from high variance
and sensitivity to hyperparameters. In contrast, value-based methods
such as Q-learning [5] and its deep variant DQN [1] remain widely used
due to their conceptual simplicity, sample efficiency in discrete
domains, and ease of implementation.

Q-learning's longevity has produced a sprawling methodological
literature. Over three decades, dozens of variants have been proposed
to address specific limitations — overestimation bias, brittle
exploration, sample inefficiency, instability under function
approximation — and many have been combined into composite agents such
as Rainbow [28]. Despite this volume of work, the *organizing
structure* through which the field is presented has remained largely
chronological or method-typed: distributional methods, ensemble
methods, replay innovations, and so on. Existing surveys [12]–[15],
as well as more recent broad-RL treatments [Ghasemi 2024/2025], have
inherited this method-type organization.

Method-type organization is bibliometrically convenient but
analytically thin. It tells the reader *what kind of thing* a method
is (an ensemble, a distributional model, a replay buffer modification)
without surfacing *why* the method exists — which weakness of vanilla
Q-learning it was designed to address, what trade-offs it introduces,
and how it compares to other approaches that target the same
weakness. As a result, readers encounter algorithms as isolated
artifacts rather than as positioned moves in an ongoing technical
conversation. Cross-method synthesis suffers accordingly.

This paper offers an alternative organization. Rather than presenting
Q-learning's variants by method type, we organize the field around
**the foundational weaknesses of vanilla Q-learning that modern
methods address**. Eight such weaknesses are identified:

1. *Overestimation bias* introduced by the `max` operator over noisy
   value estimates;
2. *Sample inefficiency* arising from uniform experience replay;
3. *Brittle exploration* under ε-greedy action selection;
4. *Reward sparsity and credit assignment* in long-horizon tasks;
5. *Distribution shift* when transferring to fixed-data (offline)
   regimes;
6. *Multi-agent coordination* failures when factoring single-agent Q
   across cooperative agents;
7. *Slow adaptation* when training per-task without transfer; and
8. *Function-approximation instability* — the "deadly triad" of
   off-policy learning, bootstrapping, and function approximation.

Each Related Works section in this paper corresponds to one of these
weaknesses. Within each section, methods are grouped by the
*mechanism* they use to address the weakness, compared head-to-head
on the trade-offs they introduce, and evaluated against the empirical
evidence relevant to that axis. The same method can appear in
multiple sections when it advances multiple axes — Rainbow [28], for
instance, recurs across five of the eight — and these cross-references
form an explicit map of the field's connective tissue.

This organization preserves all of the paper's empirical and
bibliographic contributions while reframing the surrounding analysis.
Five distinguishing contributions, summarized in Table I, support the
reframing:

1. A unified problem-axis taxonomy that integrates tabular Q-learning,
   classical deep Q-learning, and modern Q-based methods (offline RL,
   multi-agent value decomposition, distributed scaling) within a
   single framework.
2. A comparative analysis of six widely used open-source DRL
   repositories — Tianshou [6], XuanCe [7], CleanRL [8], DQN Zoo [9],
   Stable Baselines3 [10], and RLlib [11] — annotated *by axis* to
   reveal where the open-source ecosystem provides coverage and where
   it does not.
3. Curated benchmark results extracted from the Atari [2] suite as
   reported in the original papers, stratified by task category, and
   re-interpreted as evidence for or against each algorithmic axis
   rather than as a leaderboard.
4. Tabular Q-learning benchmark results from our own controlled
   reimplementations, isolating algorithmic design from architectural
   confound.
5. A thorough per-paper literature review organized by axis, with
   each method positioned by the weakness it addresses, the
   mechanism it uses, and the trade-offs it introduces.

The remainder of the paper is organized as follows. Section II
introduces the MDP formalism and formally states the eight
weaknesses that structure the Related Works. Section III describes
the methodology. Section IV — the bulk of the paper — presents the
problem-first review across the eight axes. Sections V and VI
report tabular and Atari benchmark analyses. Section VII presents
the repository comparison. Section VIII concludes with future
directions, including a planned community-maintained Q-learning
repository whose roadmap is informed by the gaps identified in
Section VII.

---

*Notes for integration:*

- The Ghasemi 2024/2025 citation needs to be added to the
  bibliography. Suggested placement: alongside existing [12]–[15].
- Table I in the current draft compares against four pre-2023
  surveys. Add Ghasemi 2024/2025 as a fifth column; the check-mark
  pattern stays favorable.
- The closing summary of section organization assumes the renumbering
  in `03-new-outline.md` (Section IV becomes the problem-axis review).
  Update accordingly if the team prefers different numbering.
- One sentence in §VII will need to distinguish our taxonomic
  repository analysis from Hundal et al. (2025)'s empirical PPO
  audit; see `07-prior-art-sweep.md`.

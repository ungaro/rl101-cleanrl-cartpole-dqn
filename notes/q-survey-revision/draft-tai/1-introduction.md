# I. Introduction {#sec-i}

Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) have
emerged as powerful paradigms for solving complex sequential
decision-making problems, with applications ranging from Atari games
[@mnih_2015_nature], [@bellemare_2013_ale] and robotic control [@ibarz_2021_robot] to autonomous driving [@inamdar_2024_safe]. Among
DRL approaches, two major families dominate: value-based methods,
which estimate action-value functions, and policy-based methods, which
directly optimize policy distributions.

While policy-based algorithms have shown strong performance in tasks
with continuous action spaces, they often suffer from high variance
and sensitivity to hyperparameters. In contrast, value-based methods
such as Q-learning [@watkins_1992_qlearning] and its deep variant DQN [@mnih_2015_nature] remain widely used
due to their conceptual simplicity, sample efficiency in discrete
domains, and ease of implementation.

Q-learning's longevity has produced a sprawling methodological
literature. Over three decades, dozens of variants have been proposed
to address specific limitations — overestimation bias, brittle
exploration, sample inefficiency, instability under function
approximation — and many have been combined into composite agents such
as Rainbow [@hessel_2018_rainbow]. Despite this volume of work, the *organizing
structure* through which the field is presented has remained largely
chronological or method-typed: distributional methods, ensemble
methods, replay innovations, and so on. Existing surveys [@urtans_2018_pygame; @jang_2019_qsurvey; @boppiniti_2021_evolution; @hafiz_2023_dqnsurvey],
as well as more recent broad-RL treatments [@ghasemi_2024_rlsurvey], have
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
3. *Brittle exploration* under $\varepsilon$-greedy action selection;
4. *Reward sparsity and credit assignment* in long-horizon tasks;
5. *Distribution shift* when transferring to fixed-data (offline)
   regimes;
6. *Multi-agent coordination* failures when factoring single-agent Q
   across cooperative agents;
7. *Slow adaptation* when training per-task without transfer; and
8. *Function-approximation instability* — the "deadly triad" of
   off-policy learning, bootstrapping, and function approximation.

Each §IV subsection corresponds to one of these
weaknesses. Within each, methods are grouped by the
*mechanism* they use to address the weakness, compared head-to-head
on the trade-offs they introduce, and evaluated against the empirical
evidence relevant to that axis. The same method can appear in
multiple sections when it advances multiple axes — Rainbow [@hessel_2018_rainbow], for
instance, recurs across five of the eight — and these cross-references
form an explicit map of the field's connective tissue.

This organization preserves all of the paper's empirical and
bibliographic contributions while reframing the surrounding analysis.
Five distinguishing contributions support the reframing:

1. A unified problem-axis taxonomy that integrates tabular Q-learning,
   classical deep Q-learning, and modern Q-based methods (offline RL,
   multi-agent value decomposition, distributed scaling) within a
   single framework.
2. A comparative analysis of six widely used open-source DRL
   repositories — Tianshou [@weng_2022_tianshou], XuanCe [@liu_2023_xuance], CleanRL [@huang_2022_cleanrl], DQN Zoo [@quan_2020_dqnzoo],
   Stable Baselines3 [@raffin_2021_sb3], and RLlib [@liang_2018_rllib] — annotated *by axis* to
   reveal where the open-source ecosystem provides coverage and where
   it does not.
3. Curated benchmark results extracted from the Atari [@bellemare_2013_ale] suite as
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
weaknesses that structure the survey. Section III describes
the methodology. Section IV — the bulk of the paper — presents the
problem-first review across the eight axes (§IV.A–H), followed by
§IV.I on recent theoretical advances and §IV.J on the emerging use
of Q-learning for foundation-model alignment.
Sections V and VI report Atari and tabular benchmark analyses.
Section VII presents the repository comparison. Section VIII
concludes with future directions, including a planned
community-maintained Q-learning repository whose roadmap is informed
by the gaps identified in Section VII. Supplementary material
provides the full per-method mapping between the problem-first axis
taxonomy of §IV and the method-type taxonomy of prior surveys,
together with notation conventions and selected derivations.

**Reading guide.** The §IV body is written for mechanism and
trade-off: each axis-section names the weakness, surveys the
solution families that respond, and compares them on the costs they
introduce. Equations in §IV are mechanism-defining rather than
fully derived; longer derivations and proof sketches are
concentrated in the supplementary material and may be skipped on a
first read without losing the structural argument. §IV.I revisits the
same ground theoretically.

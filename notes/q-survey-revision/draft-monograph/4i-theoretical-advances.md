# IV.I. Theoretical Foundations and Recent Advances {#sec-iv-i}

The eight axis-sections above survey methods responding to vanilla
Q-learning's weaknesses. This section turns to the theoretical
results that characterize *when* and *why* those responses work — and
what they provably cannot achieve. It is organized around the same
five-part template as the axis-sections: A. the theoretical landscape;
B. foundational results; C. recent advances; D. evidence-theory
interaction; E. open theoretical questions.

### A. The Theoretical Landscape

Theoretical analysis of Q-learning answers four families of question
that empirical evaluation alone cannot:

1. **Convergence.** Do the iterates of a Q-learning update converge,
   and if so, to what? Under what regime (tabular, linear
   function approximation, neural)?
2. **Sample complexity.** How many environment interactions are
   required to find an $\epsilon$-optimal Q-function with high
   probability?
3. **Approximation quality.** When Q-learning converges to a fixed
   point $Q^\dagger \neq Q^\ast$, how far is $Q^\dagger$ from
   optimal?
4. **Guarantees under distribution shift.** In offline RL, can a
   learned Q-function be provably no worse than the behavior policy?
   Under what assumptions?

Each of the eight axis-sections has a theoretical undercurrent. The
overestimation discussion in §IV.A is grounded in Jensen-type
inequalities; the deadly-triad treatment in §IV.H invokes the
classical divergence counterexamples; the offline-RL discussion in
§IV.E rests on pessimism-based suboptimality bounds. This section
brings those undercurrents together and surveys the recent
theoretical literature that interacts with each.

### B. Foundational Results

**B.1. Tabular convergence.** The canonical result for Q-learning is
the convergence theorem of [Watkins & Dayan 1992]. Under standard
stochastic-approximation conditions — each (state, action) pair
visited infinitely often, learning rates $\alpha_n$ satisfying
$\sum_n \alpha_n = \infty$ and $\sum_n \alpha_n^2 < \infty$, and
bounded rewards — the tabular Q-learning iterates converge with
probability 1 to the optimal action-value function $Q^\ast$.
The proof reduces Q-learning to a contraction in the
$\ell_\infty$ norm via the Bellman optimality operator. The result
is restricted to *finite* state-action spaces and *tabular*
representations; it does not directly apply to function-
approximation settings.

**B.2. The deadly triad.** Off-policy learning, bootstrapping, and
function approximation can collectively diverge. The classic
counterexamples are [Baird 1995] (Baird's star MDP, where the
TD iterates diverge despite the existence of an optimal
linear approximation) and [Tsitsiklis & Van Roy 1997]
(off-policy TD with linear function approximation can diverge in
finite-state MDPs). The empirical recipe of §IV.H — target networks,
replay buffers, normalization — restores stability in practice
without resolving the underlying theoretical issue: there is no
known fixed-point theorem for the combined off-policy-bootstrap-
function-approximation Q-learning operator in the general case.

**B.3. Linear function approximation with on-policy data.** When the
distribution of $(s, a)$ samples matches the policy's stationary
distribution (the *on-policy* case), the projected Bellman operator
becomes a contraction in the weighted $L^2$ norm, and TD(0) with
linear function approximation converges to a fixed point that is
near-optimal in the sense of bounded approximation error
[Tsitsiklis & Van Roy 1997]. This is the result the GTD/TDC family
of off-policy convergence algorithms [@sutton_2009_gtd] extends:
by replacing the standard TD update with a gradient on an explicit
MSPBE objective, they obtain convergence guarantees in the
off-policy linear regime — at the cost of slower empirical
convergence than vanilla TD.

### C. Recent Advances

**C.1. Distributional Bellman operator convergence.** The
distributional methods of §IV.D admit a clean theoretical account.
[@bellemare_2017_distributional] show that the distributional
Bellman operator is a $\gamma$-contraction in the maximal form of
the Wasserstein-$p$ metric — but *not* a contraction in the KL
divergence used by C51's loss. [@rowland_2018_qdistanalysis] subsequently
proved that the quantile-regression scheme of QR-DQN is a
contraction in the Wasserstein-$\infty$ metric, providing a
theoretical foundation for the practical strength of QR-DQN over
C51. The IQN and FQF generalizations [Dabney et al. 2018; Yang et
al. 2019] inherit these guarantees under suitable assumptions on
the quantile-fraction sampling distribution. The contraction
results hold for the *tabular distributional* operator; their
extension to neural function approximation remains an active area.

**C.2. Finite-time bounds for deep Q-learning.** A line of work
beginning with [@yang_2019_fqf] and [Fan et al. 2020,
*Theoretical Analysis of DQN*] derives non-asymptotic suboptimality
bounds for neural-fitted Q-iteration under specific architectural
and data-generation assumptions. The bounds have the form
$\|Q^\pi - Q^\ast\|_\infty = \tilde O\bigl(\sqrt{\epsilon_\text{stat} + \epsilon_\text{approx}}\bigr)$
where $\epsilon_\text{stat}$ is a statistical estimation error
decreasing in dataset size and $\epsilon_\text{approx}$ is the
inherent neural-network approximation error. The bounds are
loose in practice and require assumptions (e.g., bounded ReLU-network
complexity, $\beta$-mixing trajectory data) that do not hold for
realistic deep RL training pipelines, but they establish the
*shape* of guarantees one can hope for and identify the
$\epsilon_\text{approx}$ term as the dominant practical concern.

**C.3. Pessimism in offline RL.** [Jin, Yang & Wang 2021, *Is
Pessimism Provably Efficient for Offline RL?*] established the
canonical theoretical framework for offline-RL methods. The
*pessimistic value iteration* algorithm — Q-iteration with an
explicit lower-confidence-bound subtraction at each Bellman backup
— is shown to achieve $\tilde O(\sqrt{1/n})$ suboptimality without
requiring any coverage assumption on the behavior policy. This
result theoretically grounds the §IV.E methods: CQL's
conservative penalty is a tractable approximation to LCB-pessimism;
IQL's expectile regression avoids the OOD-action problem in a manner
that admits a similar pessimism interpretation. The pessimism
framework has since been extended to multi-task offline settings
[@chen_2022_offlinemulti] and to model-based offline RL [Uehara & Sun
2022], establishing offline RL as one of the most theoretically
mature axes covered in this paper.

**C.4. Stability theory under modern recipes.** The empirical
stability advances of §IV.H — layer normalization, batch
normalization, the PQN recipe — have only recently begun to admit
theoretical accounts. [@lyle_2022_capacityloss] analyze *capacity loss*
in deep RL: the phenomenon where neural Q-networks progressively
lose representational capacity over training, partly explaining
why naive deep Q-learning diverges. They show that specific
normalization schemes mitigate capacity loss in a measurable sense.
Concurrent work [Nikishin et al. 2022; *The Primacy Bias in Deep
Reinforcement Learning*] identifies a related pathology — early
training trajectories receive disproportionate optimization
attention — and propose periodic reset of network parameters as a
provable remedy under specific assumptions. The Lyle and Nikishin
threads have since converged into a recognized sub-area; [Klein et
al. 2026, *Plasticity Loss in Deep Reinforcement Learning: A
Survey*] organizes over fifty mitigation strategies — periodic
resets, regularization-to-initialization, weight-norm projections,
auxiliary tasks, replay-rate manipulation — into a unified
framework, classifying them by which neural-network failure mode
each targets. The survey makes clear that plasticity loss is now
treated as a first-class subject of deep-RL stability theory rather
than a peripheral curiosity. Together these results begin to
formalize the empirical observation that *what makes deep Q-learning
work is not what classical theory predicts*.

**C.5. Multi-agent IGM theory.** The §IV.F value-decomposition
methods rest on the Individual-Global-Max property. [Wang et al.
2020] (QPLEX) show that the IGM constraint admits a *complete*
factorization via duplex dueling, in the sense that any
IGM-compatible $Q_\text{tot}$ can be expressed in the QPLEX form.
The result establishes monotonic mixing (QMIX) as a *sufficient*
but not *necessary* condition for IGM, and provides a theoretical
ceiling on what value-decomposition methods can represent.
QTRAN's auxiliary-loss approach attempts to reach this ceiling
empirically; that it does not, in practice, identifies a
representation-vs-training-dynamics gap.

### D. Evidence–Theory Interaction

Three patterns emerge from comparing the theoretical results of §C
against the empirical findings of §IV.A–H:

- **Offline RL theory has matured rapidly.** Pessimism-based bounds
  appeared within 18 months of CQL's empirical introduction, and the
  theoretical framework now informs the design of newer methods. The
  empirical-to-theoretical lag here is short.
- **Stability theory is catching up.** The PQN recipe (§IV.H) has
  outrun its theoretical justification by several years. Recent
  capacity-loss and primacy-bias work is the field's attempt to
  formalize what empirical results have already established.
- **Distributional theory has plateaued.** The Wasserstein-contraction
  results for the tabular distributional operator are
  well-established, but extension to the neural function-approximation
  setting where distributional methods actually run has progressed
  slowly. The empirical strength of FQF over IQN, for instance, has
  no compact theoretical explanation.

The general pattern: **empirical advances precede theoretical
characterization in deep Q-learning by 2–5 years**. The recent
theoretical work surveyed here is largely catch-up rather than
forward-driving. Identifying mechanisms where theory could
*forward-drive* practical advance — for instance, exploration
strategies with provable regret bounds compatible with deep RL — is
the frontier.

### E. Open Theoretical Questions

1. **Convergence of distributional Q under function approximation.**
   The Wasserstein-contraction results of [@rowland_2018_qdistanalysis] hold in the
   tabular distributional setting; extension to the neural setting
   where C51, QR-DQN, IQN, and FQF actually run remains incomplete.
2. **Tighter finite-time bounds in the realistic deep RL regime.**
   The [@fan_2020_dqntheory] bounds require restrictive assumptions; bounds
   that match the empirical sample efficiency observed in practice
   are not known.
3. **A unified offline-online theoretical framework.** Pessimism
   guarantees the offline case; online learning has its own
   regret-bound machinery. The transition between regimes — relevant
   for offline-to-online fine-tuning (AWAC, Cal-QL) — currently
   admits only ad-hoc analyses.
4. **Multi-agent value decomposition beyond IGM.** Tasks where
   multiple optimal joint policies exist, or where execution-time
   communication is permitted, fall outside the IGM framework. A
   broader theoretical framework for cooperative multi-agent Q
   that admits these regimes is open.
5. **Capacity loss and continual stability.** The Lyle 2023 line
   formalizes one mechanism but does not yet provide
   architecture-design guidance with provable stability across the
   full training trajectory.
6. **Theoretical account of cross-axis composition.** Agent57
   composes mechanisms from §IV.B, §IV.C, and §IV.G. No theoretical
   framework currently characterizes when such compositions yield
   superadditive benefits versus when components interfere with one
   another. The Rainbow ablation provides empirical clues; theory
   has not followed.

### F. Comparison summary

| Theoretical result (year) | Regime | Mechanism | Status | Empirical anchor |
|---|---|---|---|---|
| Watkins & Dayan (1992) | Tabular | $\ell_\infty$ contraction of Bellman operator | Foundational | Tabular Q convergence — §V |
| Tsitsiklis & Van Roy (1997) | Linear FA | On-policy convergence; off-policy counterexamples | Foundational | Motivates the deadly-triad framing — §IV.H |
| Baird (1995) | Linear FA | Star-MDP divergence counterexample | Foundational | Motivates target networks — §IV.H |
| Bellemare et al. (2017) | Tabular distributional | $\gamma$-contraction in Wasserstein-$p$ | Established | C51 (§IV.D) |
| Rowland et al. (2018) | Tabular quantile | $\gamma$-contraction in Wasserstein-$\infty$ | Established | QR-DQN (§IV.D) |
| Yang et al. (2019), Fan et al. (2020) | Neural FA | Finite-time bounds under bounded-complexity assumptions | Active | DQN-family (§IV.A–C) |
| Jin, Yang, Wang (2021) | Offline | Pessimism-based suboptimality bounds | Established | CQL/IQL/PCQ (§IV.E) |
| Lyle et al. (2023) | Neural FA | Capacity-loss formalization | Active | PQN, modern stability recipes (§IV.H) |
| Nikishin et al. (2022) | Neural FA | Primacy-bias identification and reset remedy | Active | Continual stability (§IV.H) |
| Klein et al. (2026) | Neural FA | Plasticity-loss survey; >50 mitigation strategies organized | Active | Modern stability recipes (§IV.H) |
| Wang et al. (2020, QPLEX) | Multi-agent | Complete IGM-compatible representation theorem | Established | QMIX/QPLEX (§IV.F) |

The table makes the central asymmetry visible: the **classical**
results (Bellman contraction, deadly-triad counterexamples) are 25+
years old and concern regimes (tabular, linear) that today's methods
have largely outgrown. The **recent** advances (pessimism, capacity
loss, IGM theorems) have begun to close the gap, but the
neural-function-approximation regime where modern Q-learning lives
remains theoretically under-characterized. Methods that work
empirically still lack the kind of provable guarantee that, for
instance, the tabular convergence theorem provides for the
foundational case.

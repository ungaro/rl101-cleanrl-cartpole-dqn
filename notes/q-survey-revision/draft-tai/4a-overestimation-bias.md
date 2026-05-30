## IV.A. Overestimation Bias {#sec-iv-a}

**Weakness.** The update target $r + \gamma \max_{a'} Q(s',a')$ applies a
maximum over noisy estimates, which is biased upward: the expected
maximum of noisy estimators exceeds the maximum of their true means
[@smith_2006_optimizerscurse]. Propagated through the Bellman backup
the bias compounds, steering the greedy policy toward poorly-estimated
actions rather than genuinely good ones — bounded in the tabular case
but able to grow without bound under neural approximation.

**Mechanisms.** Three families debias the maximum, distinguished by
*what information they exploit*. (i) *Decoupling* separates action
selection from evaluation: tabular Double Q-learning
[@hasselt_2010_doubleq] uses two independent estimators, and Double DQN
[@hasselt_2016_doubledqn] reuses DQN's target network for evaluation at
near-zero cost, $y = r + \gamma\,Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$.
A recent re-examination, DDQL [@nagarajan_2026_ddql], shows this
target-network substitution is *not* equivalent to true two-estimator
decoupling — it carries a stale-bootstrap bias — and training two
independent networks restores lower bias and improves on 47 of 57 Atari
games under matched compute. (ii) *Ensemble* methods generalize to $K$
estimators and make the bias–variance trade-off tunable: EBQL
[@peer_2021_ebql] evaluates with the average of held-out members
($K\!\in\![5,10]$ interpolating between DQN's over- and Double DQN's
under-estimation), while REDQ [@chen_2021_redq] takes a minimum over a
random subset to deliberately induce under-estimation. (iii)
*Architectural* decomposition — Dueling DQN [@wang_2016_dueling] —
routes learning through a state-value baseline $V(s)$, reducing the
relative noise in the action-conditional term; this is primarily a
stability contribution ([§IV.H](#sec-iv-h)) that incidentally dampens
the max bias. For *hybrid* discrete–continuous actions, PDQN
[@xiong_2018_pdqn] debiases the discrete max while a learned actor emits
the continuous parameters, avoiding bias compounding across the two
levels.

**Trade-off.** No fix dominates. Double DQN trades a small
under-estimation for removing a large over-estimation at essentially no
cost; ensembles buy *calibratable* bias at $K\times$ compute and memory;
REDQ's minimum helps when the optimizer drifts toward high-$Q$ regions
but hurts when optimism is needed, as in long-horizon tasks that depend
on exploration. The right operating point therefore depends on
action-space geometry and on whether the downstream policy is more
sensitive to over- or under-estimation. Critically, these *online* fixes
transfer imperfectly to offline RL, where overestimation re-appears as
out-of-distribution action selection and demands different tools (CQL,
IQL; [§IV.E](#sec-iv-e)).

**Open questions.** The bias–variance Pareto frontier as a function of
ensemble size $K$, subset size $M$, and update rate is unmapped; and the
relationship between online debiasing and offline OOD-penalization
remains ad hoc, lacking a framework that recovers both as limits of a
single inequality.

\begin{table}[t]
\centering
\footnotesize
\caption{Overestimation-bias methods (see Section~V for benchmark
protocol caveats).}
\label{tab:axis-overest}
\begin{tabular}{@{}p{1.6cm}p{2.4cm}p{1.4cm}p{1.6cm}@{}}
\toprule
Method & Mechanism & Cost & Best at \\
\midrule
Double Q (2010) & Two swapped estimators & None & Tabular MDPs \\
Double DQN (2016) & Online selects, target evaluates & Free & Most Atari \\
DDQL (2026) & Two independent nets & $2\times$ & 47/57 Atari \\
EBQL (2021) & Avg.\ of $K{-}1$ held-out & $K\times$ & Bias calibration \\
REDQ (2021) & Min of $M$ random members & $K\times$ & Continuous control \\
Dueling$^{*}$ (2016) & $V(s)$ + centered $A(s,a)$ & Modest & Many-action states \\
\bottomrule
\end{tabular}
\\[2pt]
{\scriptsize $^{*}$Primary axis is stability (Section~IV-H); listed
here as an incidental contributor.}
\end{table}

# II. Background and Problem Setup {#sec-ii}

**MDP and Q-learning.** We adopt the standard Markov Decision Process
$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, R, P, \gamma, d_0
\rangle$: $\mathcal{S}$ and $\mathcal{A}$ are the state and action
sets, $R : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ the reward
function, $P(s' \mid s, a)$ the transition kernel, $\gamma \in [0,1)$
the discount factor, and $d_0$ the initial-state distribution. A policy
$\pi : \mathcal{S} \to \Delta(\mathcal{A})$ induces the action-value
function $Q^\pi(s,a) = \mathbb{E}_\pi\bigl[\sum_{t=0}^\infty \gamma^t
R(s_t, a_t) \mid s_0 = s, a_0 = a\bigr]$. Q-learning
[@watkins_1992_qlearning] estimates the optimal value
$Q^\ast(s,a) = \max_\pi Q^\pi(s,a)$, which satisfies the Bellman
optimality equation

$$
Q^\ast(s,a) = \mathbb{E}_{s' \sim P(\cdot \mid s,a)}\Bigl[r + \gamma \max_{a'} Q^\ast(s',a')\Bigr],
$$

via the iterative tabular update

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\Bigl(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Bigr).
$$

This update is provably convergent under tabular representation and
standard stochastic-approximation conditions [@watkins_1992_qlearning].
In practical regimes — function approximation, partial data coverage,
sparse rewards, multi-agent settings — it exhibits eight distinct
failure modes, each the organizing principle of one [§IV](#sec-iv)
subsection.

**The eight weaknesses.** *W1 — Overestimation bias:* the operator
$\max_{a'} Q(s',a')$ is biased upward whenever $Q(s',\cdot)$ carries
zero-mean noise, and recursive backups amplify the bias toward the
least accurately estimated actions ([§IV.A](#sec-iv-a)). *W2 — Sample
inefficiency:* vanilla Q-learning consumes each transition once and
uniformly, and even with experience replay [@mnih_2013_atari] uniform
sampling wastes effort on already-predicted transitions while rare
high-information ones go under-used ([§IV.B](#sec-iv-b)). *W3 — Brittle
exploration:* $\varepsilon$-greedy is dithered rather than directed and
fails to escape early plateaus when reward is sparse or delayed beyond a
random walk's reach — Montezuma's Revenge, Pitfall!, Private Eye
[@bellemare_2013_ale] ([§IV.C](#sec-iv-c)). *W4 — Reward sparsity and
credit assignment:* one-step backups propagate terminal reward slowly,
and while multi-step returns [@peng_1996_multistep] and distributional
RL [@bellemare_2017_distributional] enrich the signal, they do not by
themselves resolve what information the return carries
([§IV.D](#sec-iv-d)). *W5 — Distribution shift:* under fixed data with
narrow support (offline RL), $\max_{a'} Q(s',a')$ ranges over
unsupported actions and inherits unbounded extrapolation error
([§IV.E](#sec-iv-e)). *W6 — Multi-agent coordination:* naive per-agent
Q-learning treats co-agents as non-stationary environment while
centralized joint-action learning scales as $|\mathcal{A}|^N$, motivating
value-decomposition factorizations that preserve the
individual-global-max property ([§IV.F](#sec-iv-f)). *W7 — Slow
adaptation and sample throughput (composite axis):* we bundle two
interrelated bottlenecks — W7a, the throughput limit of a single
sequential learner, and W7b, the cost of adapting a task-specific
$Q$-function to new dynamics or rewards — because at scale their
methodological responses (distributed actor-learners, recurrent value
functions, meta-learning, predictable scaling) largely coincide, with
Agent57 the paradigm case ([§IV.G](#sec-iv-g)). *W8 —
Function-approximation instability:* the *deadly triad*
[@sutton_2018_book] of off-policy learning, bootstrapping, and function
approximation is provably unstable, so the parameterized iterates may
diverge or converge far from $Q^\ast$ ([§IV.H](#sec-iv-h)).

These failure modes are not independent — several share underlying
mechanisms — but each motivates a recognizably distinct family of
methods, and §IV's summary table consolidates their definitions
alongside the responding approaches.

**Notation.** Throughout, $s, a, r, s'$ denote a transition; $\theta$
and $\theta^-$ the parameters of an online and target network; $\pi$ a
policy; $\alpha$ a learning rate; $\gamma$ a discount factor;
$\mathcal{D}$ a replay buffer; and $\epsilon$ either an exploration rate
or a noise variable, disambiguated by context. Deviations are flagged
per section.

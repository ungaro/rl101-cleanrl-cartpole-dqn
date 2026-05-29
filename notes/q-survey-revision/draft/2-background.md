# II. Background and Problem Setup {#sec-ii}

### II.A. Markov Decision Processes

We consider the standard formulation of a Markov Decision Process
(MDP), defined as $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, R,
P, \gamma, d_0 \rangle$, where $\mathcal{S}$ denotes the set of all
possible states, and $\mathcal{A}$ the set of possible actions. The
reward function $R : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$
assigns a real-valued reward to each state-action pair. The
transition function $P(s' \mid s, a, \theta)$ specifies the
probability of transitioning to state $s'$ from state $s$ after
taking action $a$, parameterized by $\theta$. The discount factor
$\gamma \in [0,1)$ determines the importance of future rewards, and
$d_0$ denotes the initial state distribution.

A policy $\pi : \mathcal{S} \to \mathcal{A}$ (or, in the stochastic
case, $\pi : \mathcal{S} \to \Delta(\mathcal{A})$) prescribes an
action for each state. The value of a policy at state $s$ is the
expected discounted return under that policy: $V^\pi(s) =
\mathbb{E}_\pi\bigl[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) \mid s_0 =
s\bigr]$. The associated action-value function is $Q^\pi(s,a) =
\mathbb{E}_\pi\bigl[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) \mid s_0
= s, a_0 = a\bigr]$. Q-learning [5] estimates the optimal action-
value $Q^\ast(s,a) = \max_\pi Q^\pi(s,a)$ via the recursive Bellman
optimality equation,

$$
Q^\ast(s,a) = \mathbb{E}_{s' \sim P(\cdot \mid s,a)}\Bigl[r + \gamma \max_{a'} Q^\ast(s',a')\Bigr],
$$

implemented as an iterative update,

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\Bigl(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Bigr).
$$

The remainder of this paper surveys methods that modify, augment, or
replace components of this update to address one or more of the
weaknesses introduced below.

### II.B. Eight Weaknesses of Vanilla Q-Learning

The Q-learning update of §II.A is provably convergent under tabular
representation and standard stochastic approximation conditions
[Watkins & Dayan 1992]. In practical settings — function
approximation, partial coverage, sparse rewards, multi-agent
environments, fixed-data regimes — the update exhibits eight
distinct failure modes. These failure modes are not independent;
several share underlying mechanisms. But each motivates a recognizably
different family of methods, and each is the organizing principle of
one Related Works subsection.

**W1. Overestimation bias.** The operator $\max_{a'} Q(s',a')$ is
biased upward in expectation whenever $Q(s',\cdot)$ contains
zero-mean noise. Formally, for noisy estimators $\tilde Q(s',a) =
Q^\ast(s',a) + \epsilon_a$ with $\mathbb{E}[\epsilon_a] = 0$,

$$
\mathbb{E}\Bigl[\max_{a'} \tilde Q(s',a')\Bigr] \geq \max_{a'} Q^\ast(s',a'),
$$

with strict inequality whenever the noise is not degenerate
[Thrun & Schwartz 1993]. Under recursive Bellman backups this bias
propagates and amplifies, steering the greedy policy toward actions
whose values are least accurately estimated. Methods responding to
this weakness are surveyed in §IV.A.

**W2. Sample inefficiency.** Vanilla Q-learning uses each transition
once and uniformly. The introduction of experience replay [32]
allowed transitions to be reused, but uniform sampling treats all
transitions as equally informative — a transition where the agent
already predicts the outcome accurately contributes little to the
update, yet is sampled as often as a high-error transition. In
environments where high-information transitions are rare (sparse
reward, low-probability state encounters), uniform replay yields
slow convergence. Methods responding to this weakness — prioritized
sampling, learning from demonstrations, memory-efficient replay —
are surveyed in §IV.B.

**W3. Brittle exploration.** ε-greedy action selection takes a
random action with probability ε and a greedy action otherwise. This
suffices for environments where reward is sufficiently dense that
near-greedy policies explore the state space through their own
exploitation. In environments where rewards are sparse or delayed
beyond an ε-greedy random-walk's reach — Montezuma's Revenge [2],
Pitfall! [2], Private Eye [2] — ε-greedy exploration is dithered
rather than directed and fails to escape early-state plateaus. The
formal characterization is that ε-greedy is myopic with respect to
posterior uncertainty in $Q(s,a)$; methods responding to this
weakness inject structured noise, maintain posterior estimates, or
use intrinsic motivation, and are surveyed in §IV.C.

**W4. Reward sparsity and credit assignment.** When reward is
received only at the end of a long trajectory, the one-step Bellman
backup requires many iterations to propagate the signal to early
states. Multi-step returns [30] partially mitigate this by allowing
$n$-step bootstrapping:

$$
y_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n \max_{a'} Q(s_{t+n}, a').
$$

But $n$-step returns trade variance for bias and do not address the
deeper question of *what information* the return signal carries.
Distributional RL [21] reframes the question: rather than estimating
the expected return $\mathbb{E}[Z(s,a)]$, estimate the full
distribution $Z(s,a)$. The distribution carries richer credit-
assignment information — bimodality, skewness, tail behavior — that
the scalar expectation discards. Methods responding to this weakness
are surveyed in §IV.D.

**W5. Distribution shift.** Q-learning is off-policy *in principle*:
the learned $Q^\ast$ is independent of the behavior policy that
generated transitions. In practice, the off-policy guarantee is
fragile when the behavior policy is fixed and the support of the
replay distribution is narrow — the regime of offline RL. The
maximization $\max_{a'} Q(s',a')$ now ranges over actions $a'$ that
may be unsupported by the data, producing arbitrary extrapolation
errors. The failure is structural: any algorithm that backs up
through unsupported actions inherits unbounded error. Methods
responding to this weakness constrain $Q$-estimates or actions to
remain within the data support and are surveyed in §IV.E.

**W6. Multi-agent coordination.** When the environment contains
multiple cooperating agents with a shared reward, naive per-agent
Q-learning treats other agents as part of the environment — a
non-stationarity that breaks the MDP assumption. Centralized
Q-learning over the joint action space avoids this but scales
exponentially: $|\mathcal{A}|^N$ for $N$ agents. Value decomposition
methods factor $Q_\text{tot}(\mathbf{s}, \mathbf{a}) =
f(Q_1(s_1, a_1), \dots, Q_N(s_N, a_N))$ under structural constraints
on $f$ (additivity, monotonicity) that preserve the
*individual-global-max* property — that the action maximizing
$Q_\text{tot}$ jointly is the per-agent argmax of each $Q_i$.
Methods responding to this weakness are surveyed in §IV.F.

**W7. Slow adaptation.** Vanilla Q-learning trains a single
$Q$-function for a single task. Transfer to a related task —
different reward, different transition dynamics, different state
distribution — requires retraining from scratch or initialization
from a pre-trained $Q$. Neither is satisfactory when adaptation
must occur online and within a small number of episodes. Methods
responding to this weakness include meta-learning approaches that
learn an initialization or adaptation rule rather than a fixed
$Q$, distributed actor-critic architectures that share experience
across tasks, and architecturally recurrent variants that condition
on task context. They are surveyed in §IV.G.

**W8. Function-approximation instability.** The combination of
off-policy learning, bootstrapping, and function approximation —
the *deadly triad* [Sutton & Barto 2018] — is provably unstable in
the general case. Concretely, the iterates of
$Q(s,a;\theta) \leftarrow Q(s,a;\theta) + \alpha\bigl(r + \gamma
\max_{a'} Q(s',a';\theta) - Q(s,a;\theta)\bigr)$
need not converge, may diverge, and even when they converge can do
so to a point that is far from $Q^\ast$. Target networks, Polyak
averaging, layer normalization, and architectural decomposition each
address different aspects of this instability. Methods responding to
this weakness are surveyed in §IV.H.

### II.C. Notation conventions

Throughout the paper, $s, a, r, s'$ denote a transition; $\theta$ and
$\theta^-$ denote the parameters of an online and target network,
respectively; $\pi$ denotes a policy; $\alpha$ a learning rate; $\gamma$
a discount factor; $\mathcal{D}$ a replay buffer; and $\epsilon$ either
an exploration rate or a noise variable, disambiguated by context.
Deviations from this convention are explicitly flagged in each section.

# Appendix B. Notation and Selected Derivations {#sec-app-b}

This appendix consolidates symbol conventions and supplies the
derivations the main text references but does not work through. The
intent is to keep §IV bodies focused on mechanism and trade-off
while still grounding the claims for readers who want the underlying
math.

---

## B.1. Notation

The symbols below appear throughout the paper. Section-local
deviations from these conventions are flagged at first use.

| Symbol | Meaning |
|---|---|
| $s, a$ | State and action at the current time step |
| $s', a'$ | Successor state and action |
| $r$ | Immediate reward |
| $\gamma \in [0, 1)$ | Discount factor |
| $\mathcal{S}, \mathcal{A}$ | State and action spaces |
| $P(s' \mid s, a)$ | Transition probability function |
| $R(s, a)$ | Reward function (expected immediate reward) |
| $\pi$ | Policy; $\pi(a \mid s)$ for stochastic, $\pi(s)$ for deterministic |
| $V^\pi, Q^\pi$ | State-value and action-value functions under $\pi$ |
| $V^\ast, Q^\ast$ | Optimal value and action-value functions |
| $\mathcal{T}, \mathcal{T}^\ast$ | Bellman and Bellman-optimality operators |
| $\theta$ | Online network parameters |
| $\theta^-$ | Target network parameters |
| $\alpha$ | Learning rate |
| $\epsilon$ | $\varepsilon$-greedy exploration parameter; or a generic small constant |
| $\varepsilon_t$ | Noise variable at step $t$ (disambiguated by context) |
| $\mathcal{D}$ | Experience replay buffer |
| $Z(s, a)$ | Return distribution at $(s, a)$ |
| $\delta$ | Temporal-difference error |
| $K$ | Number of ensemble members |
| $\tau$ | Quantile fraction $\in [0, 1]$; or a temperature parameter |
| $\mathcal{T}_i$ | $i$-th task in a meta-learning task distribution |

Boldface lower-case (e.g., $\mathbf{a}, \mathbf{s}$) denotes joint
quantities in multi-agent settings (§IV.F).

---

## B.2. Maximum-of-noisy-estimators bound (referenced by §IV.A)

§IV.A invokes the result that the maximum of noisy estimators is
biased upward as the source of Q-learning's overestimation.
[@smith_2006_optimizerscurse] formalize this for
the related setting of decision analysis. The relevant inequality
for our purposes:

Let $\hat Q_a = Q^\ast_a + \xi_a$ for $a \in \mathcal{A}$, where
$\{\xi_a\}$ are zero-mean noise variables with positive variance. Then

$$
\mathbb{E}\Bigl[\max_{a \in \mathcal{A}} \hat Q_a\Bigr] \geq \max_{a \in \mathcal{A}} Q^\ast_a,
$$

with strict inequality when $|\mathcal{A}| \geq 2$ and at least two
$\xi_a$ have positive variance.

**Proof sketch.** For each fixed realization $\{\xi_a\}$, the
function $\xi \mapsto \max_a (Q^\ast_a + \xi_a)$ is convex in $\xi$.
Jensen's inequality applied to the expectation over $\xi$ yields

$$
\mathbb{E}_\xi\Bigl[\max_a (Q^\ast_a + \xi_a)\Bigr] \geq \max_a \mathbb{E}_\xi[Q^\ast_a + \xi_a] = \max_a Q^\ast_a.
$$

Strict inequality follows from non-degeneracy of the maximum under
random perturbation. Iterated through the Bellman backup, the bias
compounds, motivating the decoupling methods of §IV.A.B.1 and the
ensemble methods of §IV.A.B.2.

---

## B.3. Categorical distributional projection (referenced by §IV.D)

C51 [@bellemare_2017_distributional] maintains a return distribution as a categorical
distribution over $N = 51$ fixed atoms in $[V_\text{min},
V_\text{max}]$. After applying the distributional Bellman update
$\hat{\mathcal{T}} Z(s, a) = r + \gamma Z(s', a^\ast)$, the
resulting distribution generally has support outside the fixed atom
set; the projection operator $\Phi$ maps it back.

**Definition.** Let $\{z_i\}_{i=0}^{N-1}$ be the atom locations with
$z_i = V_\text{min} + i \Delta z$, $\Delta z = (V_\text{max} - V_\text{min}) / (N-1)$.
Given a candidate distribution with probability mass $p_j$ at
locations $\tilde z_j$, the projection $\Phi$ produces a distribution
on $\{z_i\}$ with probabilities

$$
\bigl(\Phi(\tilde z, p)\bigr)_i = \sum_j p_j \cdot \max\Bigl(0, 1 - \frac{|\tilde z_j - z_i|}{\Delta z}\Bigr),
$$

with $\tilde z_j$ clipped to $[V_\text{min}, V_\text{max}]$ before
projection. Probability mass is conserved by construction (the
weights of any $\tilde z_j$ sum to 1 across the two adjacent atoms).

The KL loss in C51's training objective is then

$$
\mathcal{L}_\text{C51} = \mathrm{KL}\bigl(\Phi(\hat{\mathcal{T}} Z_{\bar\theta}) \,\|\, Z_\theta\bigr),
$$

where $Z_\theta$ is the predicted distribution under the online
parameters and $\bar\theta$ are the target parameters. Note that
the projection step is *non-contractive* in the KL norm — the
contraction results of §IV.I.C.1 hold in the Wasserstein metric
instead.

---

## B.4. Wasserstein contraction of the distributional Bellman operator (referenced by §IV.I.C.1)

[@bellemare_2017_distributional] establish that the distributional
Bellman operator $\mathcal{T}_\pi^d$ is a $\gamma$-contraction in
the maximal form of the Wasserstein-$p$ distance over return
distributions.

**Statement.** For two return distribution functions
$Z_1, Z_2: \mathcal{S} \times \mathcal{A} \to \mathscr{P}(\mathbb{R})$,

$$
\bar W_p\bigl(\mathcal{T}_\pi^d Z_1, \mathcal{T}_\pi^d Z_2\bigr) \leq \gamma \bar W_p\bigl(Z_1, Z_2\bigr),
$$

where $\bar W_p(Z_1, Z_2) = \sup_{s, a} W_p\bigl(Z_1(s, a), Z_2(s, a)\bigr)$
and $W_p$ is the Wasserstein-$p$ distance between scalar
distributions.

**Proof sketch.** The distributional Bellman operator decomposes
into three steps:
(i) sample a transition $(s, a, r, s')$ from $P$;
(ii) shift and scale the return distribution at the next state
by the affine map $z \mapsto r + \gamma z$;
(iii) integrate over the policy $\pi(a' \mid s')$.

Wasserstein-$p$ is preserved under expectation (step iii) and
satisfies $W_p(\delta_r + \gamma X, \delta_r + \gamma Y) = \gamma W_p(X, Y)$
under affine transformations (step ii). Taking the supremum over
$(s, a)$ on both sides yields the stated contraction.

The KL divergence used by C51's loss does *not* satisfy an analogous
contraction; this is the technical motivation for QR-DQN's switch to
quantile-regression losses, where the Wasserstein-$\infty$
contraction is preserved [@rowland_2018_qdistanalysis].

---

## B.5. Pessimism lower-bound argument for offline RL (referenced by §IV.E and §IV.I.C.3)

Pessimistic value iteration constructs a lower confidence bound
(LCB) on Q-values at each Bellman backup. The canonical version due
to [@jin_2021_pessimism] runs

$$
\hat Q^{k+1}(s, a) = r(s, a) + \gamma \sum_{s'} \hat P(s' \mid s, a) \max_{a'} \hat Q^k(s', a') - b(s, a),
$$

where $b(s, a)$ is a *bonus* term inversely proportional to the
visitation count $n(s, a)$ in the offline dataset. Concretely,
$b(s, a) = c \sqrt{1/n(s, a)}$ for a problem-dependent constant
$c$.

**Suboptimality bound.** Let $\hat \pi$ be the policy greedy with
respect to the fixed point $\hat Q^\ast$ of the LCB iteration, and
let $\pi$ be any policy that the offline dataset supports. Then

$$
V^\pi(s_0) - V^{\hat \pi}(s_0) \leq \tilde O\Bigl(\sqrt{\tfrac{1}{n}}\Bigr),
$$

where $n$ is the total dataset size, $s_0$ is the initial state, and
the $\tilde O$ hides logarithmic factors and the problem horizon.

**The key insight.** Pessimism makes $\hat Q^\ast(s, a)$ a
*lower bound* on $V^\pi$ for any supported policy. The greedy policy
with respect to $\hat Q^\ast$ therefore competes with the best
*supported* policy, not with the unrestricted optimum. Crucially,
this bound holds *without coverage assumptions* on the behavior
policy — it adapts gracefully when the dataset covers only a small
subset of the state space.

CQL's [@kumar_2020_cql] conservative penalty is a tractable
relaxation of this LCB construction: rather than maintaining
explicit confidence bounds, CQL adds a regularizer that drives
$\hat Q$ down at OOD actions, achieving the same pessimism guarantee
under specific assumptions on the regularization weight $\alpha$.
IQL's expectile regression achieves a related effect via the
asymmetric quantile loss without explicit OOD-action sampling.

---

## B.6. QPLEX IGM completeness (referenced by §IV.F and §IV.I.C.5)

The IGM (Individual-Global-Max) constraint requires that the joint
argmax of $Q_\text{tot}$ coincide with the per-agent argmaxes of
$\{Q_i\}$:

$$
\arg\max_{\mathbf{a}} Q_\text{tot}(\mathbf{s}, \mathbf{a}) = \bigl(\arg\max_{a_1} Q_1, \dots, \arg\max_{a_N} Q_N\bigr).
$$

**QMIX's sufficient condition.** Monotonic mixing — $\partial Q_\text{tot} / \partial Q_i \geq 0$ for all $i$ — implies IGM but is not necessary. There exist IGM-compatible
$Q_\text{tot}$ that QMIX cannot represent.

**QPLEX's claim** [@wang_2021_qplex]. Decomposing $Q_\text{tot} =
\sum_i V_i(\tau_i) + A_\text{tot}(\mathbf{s}, \mathbf{a})$ with
$A_\text{tot}$ produced by a duplex-dueling mixer admits *every*
IGM-compatible $Q_\text{tot}$. The proof constructs an explicit
duplex-dueling representation for any IGM-satisfying joint
$Q_\text{tot}$ by reducing to per-agent advantage representations
that satisfy IGM individually.

**Significance.** The result identifies QPLEX's representational
capacity as the ceiling of value-decomposition methods. QTRAN's
auxiliary-loss approach reaches the same ceiling via a different
construction but at the cost of training instability discussed in
§IV.F.B.4. The representation–training-dynamics gap that QTRAN
exposes is precisely the open question listed in §IV.I.E item 4.

---

## B.7. Tabular Q-learning convergence (referenced by §V and §IV.I.B.1)

The canonical Watkins & Dayan (1992) convergence proof for tabular
Q-learning runs as follows.

**Setting.** A finite MDP $\langle \mathcal{S}, \mathcal{A}, P, R,
\gamma \rangle$ with bounded rewards. The Q-learning update at
step $t$ is

$$
Q_{t+1}(s_t, a_t) = (1 - \alpha_t(s_t, a_t)) Q_t(s_t, a_t) + \alpha_t(s_t, a_t) \bigl(r_t + \gamma \max_{a'} Q_t(s_{t+1}, a')\bigr).
$$

**Convergence conditions.**

1. **Coverage:** Every $(s, a) \in \mathcal{S} \times \mathcal{A}$ is
   visited infinitely often.
2. **Learning rate decay:** $\sum_t \alpha_t(s, a) = \infty$ and
   $\sum_t \alpha_t(s, a)^2 < \infty$ for every $(s, a)$.
3. **Bounded rewards:** $|r_t| \leq R_\text{max}$ for all $t$.

**Theorem.** Under conditions 1–3, $Q_t \to Q^\ast$ with probability
1.

**Proof outline.** Define the Bellman optimality operator
$(\mathcal{T}^\ast Q)(s, a) = R(s, a) + \gamma \mathbb{E}_{s'}[\max_{a'} Q(s', a')]$.
The Q-learning update can be rewritten as a stochastic approximation
to the fixed-point iteration $Q \leftarrow \mathcal{T}^\ast Q$.
Since $\mathcal{T}^\ast$ is a $\gamma$-contraction in the
$\ell_\infty$ norm and has unique fixed point $Q^\ast$, classical
stochastic approximation theory [Robbins-Monro 1951; Tsitsiklis 1994]
yields almost-sure convergence under conditions 1–3.

**Limitations.** The proof requires *tabular* representation: each
$Q(s, a)$ is a separately-updated scalar, and the contraction
argument uses the unique-fixed-point property of $\mathcal{T}^\ast$.
Function approximation breaks both requirements — updates at one
$(s, a)$ now affect others, and the Bellman operator composed with
function-approximation projection is not in general a contraction.
This is the foundational source of the deadly-triad instability of
§IV.H.A and §IV.I.B.2.

---

## B.8. Pointers for further reading

The derivations above are the ones most directly load-bearing for
the claims in §IV. Readers seeking deeper treatment are referred to
the following canonical sources:

- *Tabular Q-learning convergence and TD methods:* Sutton & Barto
  2018, *Reinforcement Learning: An Introduction* (2nd ed.),
  chapters 6 and 7.
- *Function approximation and the deadly triad:* Sutton & Barto
  2018, chapter 11; Tsitsiklis & Van Roy 1997.
- *Distributional RL theory:* Bellemare, Dabney & Rowland 2023,
  *Distributional Reinforcement Learning* (MIT Press).
- *Offline RL theory and pessimism-based methods:* Levine et al.
  2020 survey; Jin, Yang & Wang 2021.
- *Multi-agent value decomposition:* Wang et al. 2020 (QPLEX);
  Albrecht & Stone 2018, *Multiagent Learning* (recent survey).

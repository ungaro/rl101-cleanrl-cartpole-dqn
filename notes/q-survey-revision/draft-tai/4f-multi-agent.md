## IV.F. Multi-Agent Coordination {#sec-iv-f}

**Weakness.** Cooperative multi-agent RL models $N$ agents in a
shared-reward dec-POMDP where each agent acts on a local
observation-action history $\tau_i$ but the reward depends on the joint
action. Independent Q-learning treats co-learners as non-stationary
dynamics and destabilizes, while a centralized joint $Q_\text{tot}$ over
the product action space scales exponentially in $N$. The crux is the
**Individual-Global-Max (IGM)** property: a factorization
$Q_\text{tot}=f(Q_1,\dots,Q_N)$ must let each agent act greedily on its
local $Q_i$ yet recover the joint greedy action,
$\arg\max_{\mathbf{a}}Q_\text{tot} = \bigl(\arg\max_{a_1}Q_1,\dots,\arg\max_{a_N}Q_N\bigr)$,
so that decentralized execution stays globally optimal.

**Mechanisms.** Decomposition families trade representational capacity
against training stability by how strictly they constrain $f$. (i)
*Additive* — VDN [@sunehag_2017_vdn] sets
$Q_\text{tot}=\sum_i Q_i(\tau_i,a_i)$, trivially IGM-satisfying and
trivial to train, but unable to express non-separable coordination. (ii)
*Monotonic* — QMIX [@rashid_2018_qmix] relaxes additivity to
$\partial Q_\text{tot}/\partial Q_i \geq 0$ via a state-conditioned
hypernetwork mixer, preserving IGM with richer capacity and becoming the
dominant baseline. (iii) *Full-expressiveness via dueling* — QPLEX
[@wang_2021_qplex] mixes per-agent advantage streams under an
IGM-preserving constraint, representing the entire IGM class. (iv)
*Constraint-free* — QTRAN [@son_2019_qtran] learns an unconstrained joint
$Q$ and enforces IGM through auxiliary losses, maximal in principle but
hard to optimize.

**Trade-off.** The progression VDN → QMIX → QPLEX → QTRAN raises
representational capacity monotonically, but capacity and stability are
*not* independent: the empirical SMAC ordering is QMIX ≈ QPLEX > VDN >
QTRAN, because structural constraints that buy expressiveness also degrade
training dynamics. QPLEX's full expressiveness pays off only on
super-hard scenarios where representation is the bottleneck, whereas
QTRAN's theoretical maximum never realizes — its loss-enforced IGM is the
canonical case of capacity without trainability. QFIX/Q+FIX [@baisero_2025_qfix]
reframes this tension as a *base-plus-correction* pattern rather than a
monolithic mixer redesign: a small per-agent residual network corrects a
VDN, QMIX, or QPLEX base toward the full IGM class, recovering the QPLEX
expressiveness ceiling with simpler training dynamics and consistent gains
over both QMIX and QPLEX on SMACv2 and Overcooked. This decouples the two
axes that the earlier methods conflated. These instabilities compound the
single-agent biases of [§IV.A](#sec-iv-a) and [§IV.H](#sec-iv-h), and the
required centralized critic limits the framework to settings where
centralized training is feasible.

**Open questions.** The IGM frame assumes a single optimal joint policy
and greedy decentralized execution, leaving equilibrium selection,
communication, and heterogeneous agents to ad-hoc extensions. Multi-agent
off-policy correction [@yu_2021_imp] and the offline multi-agent regime —
where per-agent behavior constraints need not yield coordinated joint
behavior — remain among the cleanest cross-axis open problems.

| Method (year) | Representational capacity | Training stability | Core mechanism |
|---|---|---|---|
| VDN (2018) | Lowest (additive only) | Highest | $Q_\text{tot}=\sum_i Q_i$ |
| QMIX (2018) | Medium (monotonic) | High | State-conditioned monotonic hypernetwork mixer |
| QPLEX (2020) | High (full IGM class) | Medium | Duplex dueling on advantage streams |
| QTRAN (2019) | Maximum (unconstrained) | Lowest | Unconstrained joint $Q$ + auxiliary IGM losses |
| QFIX/Q+FIX (2025) | High (recovers IGM ceiling) | High | Per-agent residual correction on a base mixer |

: Value-decomposition methods.

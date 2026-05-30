## IV.C. Brittle Exploration {#sec-iv-c}

**Weakness.** $\varepsilon$-greedy takes a uniform-random action with
probability $\varepsilon$ and otherwise acts greedily — *dithered* noise
applied independently per step, with no memory of past exploration and no
model of which actions remain uncertain. It is myopic with respect to
epistemic uncertainty: a state-action pair visited many times is sampled
at the same rate as one visited never, so it fails when reward is sparse,
delayed, or behind passages whose random-walk hitting time is exponential
in trajectory length — the canonical case being Montezuma's Revenge
[@bellemare_2013_ale], where DQN scores essentially zero.

**Mechanisms.** Four families induce a posterior over $Q(s,a)$ or an
exploration bonus, distinguished by *what they exploit*. (i) *Noise
injection* perturbs the network rather than the action: parameter-space
noise [@plappert_2018_paramnoise] adds $\mathcal{N}(0,\sigma^2 I)$ to
$\theta$ once per episode (with layer normalization [@ba_2016_layernorm]
and adaptive $\sigma$) for temporally consistent exploration, while
NoisyNet [@fortunato_2018_noisynet] replaces weights with learned
$\mu+\sigma\odot\varepsilon$ sampled per forward pass so the network
*learns* where to keep exploring — a load-bearing component of Rainbow
[@hessel_2018_rainbow]. (ii) *Ensemble disagreement* treats variance across
$K$ Q-heads as epistemic signal: Bootstrapped DQN [@osband_2016_bootstrapped]
samples one bootstrap-trained head per episode, and UCB Q-Ensembles
[@chen_2017_ucbq] act on $\arg\max_a(\mu(s,a)+\lambda\sigma(s,a))$ — the
same ensemble that serves bias control in [§IV.A](#sec-iv-a). (iii)
*Belief / posterior modulation* reweights backups: CBDQ [@zhao_2025_cbdq]
maintains a clustered belief $b_t(a\mid s)$ that weights future values,
while Posterior Sampling DQN [@sokar_2023_psdqn] draws a $Q$-hypothesis
from an approximate posterior [@dearden_1998_bayesianq] per episode (deep
Thompson sampling). (iv) *Intrinsic motivation* modifies reward: RND
[@burda_2019_rnd] adds a bonus from random-network prediction error that
decays as states become familiar, the first method to reach non-trivial
Montezuma scores; Go-Explore [@ecoffet_2021_goexplore] instead archives
visited states and *returns-then-explores*, decoupling reaching a state
from exploring it, and fully solves Montezuma's Revenge.

**Trade-off.** No method dominates. Per-episode perturbation (parameter
noise, Bootstrapped DQN) yields multi-step consistency that solves
partially-observed tasks like flickering Pong [@hausknecht_2015_drqn],
whereas NoisyNet's per-pass sampling adapts faster but loses that
property. Ensembles multiply forward-pass cost and memory by $K$ (largely
hidden by batching). Intrinsic bonuses add a scale hyperparameter and
risk novelty-addiction that crowds out extrinsic reward. Go-Explore is
empirically dominant on hard-exploration Atari but introduces a
non-Markovian archive whose extension to continuous spaces is unsettled.
Crucially, demonstrations remain the most effective sidestep on the
hardest games — Rainbow's best honest Montezuma result trails DQfD by an
order of magnitude — sharpening the exploration-vs-demonstration tension
of [§IV.B](#sec-iv-b); distributional gains ([§IV.D](#sec-iv-d)) do not
transfer to this axis at all.

**Open questions.** All methods here fix one exploration strategy for
all of training, yet game-by-game variance argues for adaptive selection
(meta-learned, bandit-controlled as in Agent57 [@badia_2020_agent57]
[§IV.G](#sec-iv-g), or uncertainty-thresholded), and no consensus exists
on balancing intrinsic against extrinsic reward or on archive-based
exploration in continuous latent spaces.

| Method (year) | Mechanism | Cost | Best at |
|---|---|---|---|
| Param Space Noise (2017) | Per-episode $\theta$ perturbation | Layer norm | Episode-consistent exploration |
| NoisyNet (2018) | Per-pass weight noise, learned $\sigma$ | 2× forward | Adaptive exploration |
| Bootstrapped DQN (2016) | $K$-head ensemble, head per episode | $K\times$ memory | Episode-coherent exploration |
| UCB Q-Ensemble (2018) | $\mu+\lambda\sigma$ over $K$-ensemble | $K\times$ compute | Uncertainty-aware action |
| CBDQ (2025) | Belief over actions weights backups | Clustering | Classic control + driving |
| Posterior Sampling DQN (2023) | Sample $Q$ from posterior per episode | Approx. posterior | Cyclic environments |
| RND (2018) | Random-network distillation bonus | Bonus scale | Sparse-reward Atari |
| Go-Explore (2019/21) | Archive + return-then-explore | State hashing | Hardest Atari exploration |

: Exploration methods.

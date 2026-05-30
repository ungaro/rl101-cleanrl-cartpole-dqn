## IV.G. Scaling and Slow Adaptation {#sec-iv-g}

**Weakness.** This axis is *composite*: vanilla Q-learning is both
sample-throughput-bound (W7a) and slow to adapt across tasks (W7b). A
single sequential learner cannot consume transitions faster than the
environment produces them — collecting Rainbow's [@hessel_2018_rainbow]
200M Atari frames takes weeks of wall-clock time — while a Q-function
trained on task $\mathcal{T}_1$ specializes to $\mathcal{T}_1$ and
neither retraining nor warm-starting from $Q^{\mathcal{T}_1}$ yields
few-shot transfer to a related $\mathcal{T}_2$. The two failure modes
are bundled because the methods that answer either mostly answer both.

**Mechanisms.** The W7a (throughput) cluster parallelizes interaction.
Ape-X [@horgan_2018_apex] decouples collection from learning — many
actors feed a centralized prioritized buffer that one GPU learner drains
— for $\sim\!50\times$ wall-clock speedup; R2D2 [@kapturowski_2019_r2d2]
adds an LSTM agent with replay *burn-in* to re-sync stale recurrent
states, building on DRQN's [@hausknecht_2015_drqn] recurrent head for
partial observability; Agent57 [@badia_2020_agent57] adds an NGU
intrinsic-motivation module and a bandit over an exploration-discount
policy portfolio, the first to clear human level on all 57 games. PQN
[@gallici_2024_pqn] instead uses synchronous vectorized environments,
trading actor-learner staleness for single-machine simplicity
([§IV.H](#sec-iv-h)). The W7b (adaptation) cluster meta-trains across a
task distribution $p(\mathcal{T})$, differing in *what is meta-learned*:
an initialization (MAML-Q [@finn_2017_maml; @mendonca_2019_metaq], the
first-order Reptile-Q [@nichol_2018_reptile], proximal ProMP
[@rothfuss_2019_promp])
$$\theta_0^\ast = \arg\min_{\theta_0} \mathbb{E}_{\mathcal{T}}\bigl[\mathcal{L}_{\mathcal{T}}(\theta_0 - \alpha \nabla_{\theta_0} \mathcal{L}_{\mathcal{T}}(\theta_0))\bigr];$$
a context embedding $z$ that conditions $Q(s,a,z)$ with no gradient
adaptation (PEARL [@rakelly_2019_pearl], off-policy MQL
[@fakoor_2020_mql]); or a forward-pass update rule, where a transformer
reads the recent trajectory and predicts actions in-context — algorithm
distillation and AdA, consolidated since 2023 into in-context Q-learning
proper (SICQL [@liu_2026_sicql], compositional ICQL [@xu_2026_icql]).

**Trade-off.** The deep distinction is between *hardware-driven*
throughput and *algorithmic* few-shot adaptation. W7a methods buy
performance with parallelism and compute: the Ape-X→R2D2→Agent57
progression is the single largest source of Atari gains since DQN,
suggesting throughput is the binding constraint at scale and that
algorithmic axes become secondary once compute is abundant. W7b methods
instead buy data-efficiency on new tasks — adapting in 5–50 episodes
where single-task baselines need thousands — at 10–100× meta-training
cost whose payoff is empirically mixed. Methods span both because the
distributed compute W7a developed is precisely what makes meta-training
and large in-context models feasible, and because *implicit*
meta-learning (a single Agent57/AdA-style agent trained across the whole
distribution) often matches *explicit* outer-loop meta-learners while
also exploiting the distributed actor pool. Predictable scaling laws
[@rybkin_2025_scalinglaws] reframe both: the data-versus-compute Pareto
follows the updates-to-data ratio, so the optimal allocation can be
extrapolated from a small pilot — orthogonal to, and compatible with,
the architecture that supplies the throughput.

**Open questions.** Whether meta-Q methods extend beyond narrow
parametric task distributions to heterogeneous state/action/reward
families is unresolved, and no compute-matched, axis-stratified ablation
exists to separate genuine algorithmic gains from gains that are merely
compensating for scale.

| Method (year) | Sub-axis (W7a/W7b) | Mechanism | Best at |
|---|---|---|---|
| DRQN (2015) | W7b | LSTM head over DQN | Partial observability |
| Ape-X (2018) | W7a | Async actors + central learner + PER | 50x wall-clock speedup |
| R2D2 (2019) | W7a | Ape-X + LSTM + replay burn-in | Memory-demanding tasks |
| Agent57 (2020) | W7a/W7b | R2D2 + NGU + bandit policy portfolio | All 57 Atari at human level |
| PQN (2024) | W7a | Synchronous vectorized envs + LayerNorm | Compute-efficient single machine |
| MAML-Q / Reptile-Q / ProMP (2017+) | W7b | Meta-learn fast-adapt initialization | Few-shot gradient transfer |
| PEARL / MQL (2019/20) | W7b | Inferred task-context embedding | Adaptation without gradient steps |
| SICQL / ICQL (2026) | W7b | In-context Q via transformer forward pass | Compositional task transfer |

: Distributed-scaling and adaptation methods.

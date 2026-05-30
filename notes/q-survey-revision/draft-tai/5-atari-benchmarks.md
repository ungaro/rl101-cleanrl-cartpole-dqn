# V. Atari Benchmark Analysis {#sec-v}

We read the historical Atari record diagnostically rather than as a
leaderboard. Games are stratified by task category — reaction-time /
dense-reward, strategic-planning, sparse-reward, and large-observation
— and each category preferentially stresses one or two of the
weakness axes of §IV: reaction-time and large-observation games load
function-approximation stability (W8), strategic-planning games load
credit assignment (W4), and sparse-reward games load brittle
exploration (W3). The question is therefore not which method scores
highest, but whether methods targeting a given weakness excel on its
diagnostic category and sit near baseline elsewhere — the central
empirical prediction of the taxonomy. Scores are extracted from the
original publications under each paper's own protocol; the full
per-game tables (supplementary material) report the raw numbers, and we
draw only category-level patterns from them here. The aim is to test
predictions, not to crown a method, and where a pattern could equally
be explained by reporting conventions we say so.

**Sparse-reward games discriminate exploration, where the axis matters
most.** On Montezuma's Revenge, Pitfall!, and Private Eye, methods that
target exploration explicitly — NoisyNet, Bootstrapped DQN, posterior
sampling — produce nonzero Montezuma scores where vanilla DQN and
methods aimed at other axes do not, and Rainbow's small Montezuma score
is attributable to its NoisyNet component. Demonstration-based DQfD
scores far higher still, illustrating the trade-off of
[§IV.B](#sec-iv-b): demonstrations substitute for directed exploration.
Crucially, intrinsic motivation is necessary but not sufficient — the
category is not saturated by the methods tabulated here, since RND and
Go-Explore ([§IV.C](#sec-iv-c)) report Montezuma scores one to two
orders of magnitude larger again.

**Strategic-planning games favor distributional methods.** On
credit-assignment-heavy games such as Q*bert and Ms. Pac-Man,
quantile and distributional methods (QR-DQN, IQN, FQF) produce the
category's most striking results — QR-DQN's Q*bert score is the
single largest in the compilation, far above the best
non-distributional result on that game — consistent with modeling the
full return distribution — rather than only its mean — helping
propagate value over the long action chains these games reward
([§IV.D](#sec-iv-d)). The magnitude of the gap warrants caution: such
extreme scores can reflect a handful of favorable seeds as much as a
mechanism, so we treat the *consistency* of the distributional advantage
across the category, not any single number, as the signal. Prioritized
replay shows the same category profile, strongest exactly where
high-information transitions are rare and uniform sampling wastes
updates.

**Dense-reward and large-observation games show that stability, not
capability, is the bottleneck.** On reaction-time control (Breakout,
Space Invaders, Enduro) and large-observation games (River Raid, Hero),
inter-method variance is comparatively small: Double DQN, Dueling DQN,
Rainbow, and distributional methods cluster within roughly a factor of
two. Here overestimation control buys only modest gains over the DQN
baseline: when rewards are frequent the agent receives enough signal
that the binding constraint is keeping training stable rather than
extracting value from rare events. The weakness signal is therefore
function-approximation stability ([§IV.H](#sec-iv-h)) rather than
capability, and PQN's strong showing — from a deliberately simple
parallelized Q-learner — is consistent with that reading: where
stability is the issue, architectural restraint suffices and elaborate
debiasing or distributional machinery adds little.

These patterns are suggestive, not conclusive. The original-paper
numbers are point estimates over three to five seeds without confidence
intervals, gathered under heterogeneous no-op-start, sticky-action, and
frameskip protocols, so apparent gaps may reflect protocol or compute
differences rather than algorithmic superiority.
[@agarwal_2021_rliable] document how this standard practice
systematically misrepresents performance, and introduce the *rliable*
framework, which substitutes robust aggregate metrics (interquartile
mean, optimality gap, probability of improvement) and stratified
bootstrap intervals for point-estimate comparisons. The
deterministic-by-default Arcade Learning Environment
[@bellemare_2013_ale; @machado_2018_aleeval] compounds the concern,
since methods exploiting frame-perfect sequences can score artificially
well. We therefore refrain from declaring per-game winners and read the
category narrative as the load-bearing evidence.

Atari also leaves capabilities undiagnosed, motivating newer
benchmarks: ProcGen [@cobbe_2020_procgen] measures generalization
across procedural variation, NetHack [@kuttler_2020_nethack] tests
exploration and credit assignment at extreme horizons, BSuite
[@osband_2020_bsuite] isolates single capabilities in an explicitly
axis-stratified manner, D4RL [@fu_2020_d4rl] diagnoses offline
distribution shift ([§IV.E](#sec-iv-e)), and SMAC
[@samvelyan_2019_smac] targets multi-agent coordination
([§IV.F](#sec-iv-f)).

| Task category | Diagnoses (axis) | Methods that excel | Representative finding |
|---|---|---|---|
| Reaction-time / dense reward | Function-approx stability (W8) | Double DQN, Dueling, PQN | Methods cluster within ~2x; overestimation control buys modest gains |
| Strategic planning | Credit assignment (W4) | QR-DQN, IQN, FQF | Distributional methods dominate Q*bert / Ms. Pac-Man |
| Sparse reward | Brittle exploration (W3) | NoisyNet, Bootstrapped DQN, DQfD, RND/Go-Explore | Intrinsic motivation necessary but not sufficient on Montezuma |
| Large observation | Function-approx stability (W8) | Dueling, distributional, PQN | Low inter-method variance; capability not the bottleneck |

: Atari task categories as axis diagnostics.

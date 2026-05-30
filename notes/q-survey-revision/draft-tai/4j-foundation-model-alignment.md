## IV.J. Q-Learning for Foundation Model Alignment {#sec-iv-j}

Aligning large language and vision-language models recasts a
familiar problem in unfamiliar dimensions, and that change of scale
is what turns it into a Q-learning problem. Each generated token is
an action drawn from a vocabulary of $10^4$–$10^5$ entries, so a
single response is a long trajectory through an enormous discrete
action space. The reward is sequence-level and sparse: an entire
generation receives one scalar from a learned reward model, with no
built-in credit assignment to the tokens that earned it — a
sparse-reward problem ([§IV.D](#sec-iv-d)) at extreme scale.
Finally, alignment must not destroy the pretrained model's fluency,
so the objective is regularized by a KL penalty against a reference
policy, replacing the usual return with a return-minus-KL target:

$$
Q_\text{reg}(s,a) = r(s,a) + \gamma\,\mathbb{E}\big[V_\text{reg}(s')\big] - \beta \log \pi_\text{ref}(a\mid s).
$$

Methods cluster by how they exploit this structure. Offline
trajectory and value methods reuse fixed data, including failed
attempts: Q-Transformer [@chebotar_2023_qtransformer] trains an
autoregressive transformer Q-network over discretized action tokens
with conservative penalties ([§IV.E](#sec-iv-e)), and VLM
Q-Learning carries the same off-policy recipe to multimodal
(image, text, action) data. A second group reinterprets the model's
own machinery: ShiQ treats per-token logits as Q-values up to a
bias correction, making temporal-difference learning directly
compatible with the KL-regularized objective and removing the
online rollouts that policy-gradient RLHF requires. A third,
Q-sharp (Q♯), targets provable behavior by guiding the reference
policy with a *distributional* Q-function, trading lower KL for a
given reward gain. In-context variants — SICQL and ICQL — fit
transformer Q-functions that adapt at inference time with no
gradient updates, and reward-shaping approaches such as Q-shaping
inject value estimates as auxiliary shaping signal rather than as
the optimization target.

This is frontier, fast-moving work: the benchmarks are immature
(reward models judging reward models), the methods are months apart,
and value-based alignment may eventually fold into a broader axis
on reward-model robustness. Its long-term place in the taxonomy is
not yet settled, but its present importance to the field is.

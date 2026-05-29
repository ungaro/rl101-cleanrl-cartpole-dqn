# III. Methodology

This review surveys Q-learning algorithms spanning over three decades
of development, from foundational tabular approaches to recent deep
variants. We include influential papers that shaped the theoretical
landscape, introduced practical improvements, or demonstrated wide
applicability in real-world domains. The selection prioritizes
methods whose contributions can be mapped onto one or more of the
eight weakness axes introduced in §II.B — methods that respond to a
recognizable weakness of vanilla Q-learning with a distinct
mechanism.

### A. Source selection

We surveyed the value-based RL literature published between 1989 and
2025, with primary attention to peer-reviewed venues (NeurIPS, ICML,
ICLR, JMLR, *Nature*, IEEE journals) and to widely-cited preprints
that have demonstrably shaped subsequent work. Five inclusion
criteria were applied:

1. **Method centrality** — the work introduces a methodological
   contribution to the Q-learning family rather than applying
   Q-learning to a domain.
2. **Mechanism distinctness** — the contribution is mechanistically
   distinguishable from prior work along at least one of the eight
   axes of §II.B.
3. **Empirical validation** — the work reports results on a
   recognized benchmark (Atari, classic control, D4RL, SMAC, or
   equivalent).
4. **Cross-reference impact** — the work is cited as foundational by
   subsequent methods in the same axis-family.
5. **Reproducibility** — code is publicly available or the algorithm
   is sufficiently specified to be independently re-implemented.

Methods satisfying all five criteria received per-paper treatment.
Methods satisfying a subset received compact treatment as part of a
family (e.g. the dueling-mixing family in §IV.F is treated through
its canonical members VDN, QMIX, QPLEX, QTRAN).

### B. Comparison with prior surveys

To position this survey against existing work, we compiled Table I,
which compares this paper with five prior Q-learning–focused surveys
[12]–[15] and [Ghasemi 2024/2025] along five distinguishing
dimensions: analysis of public DQN repositories, unified taxonomy
spanning tabular and deep Q-learning, extraction of original-paper
Atari benchmarks, classic control benchmarks from controlled
re-implementation, and per-paper review organized around mechanism
and axis. Where prior surveys focus on chronological or method-type
organization, this paper's problem-first axis structure (§IV) is, to
our knowledge, the first such treatment of Q-learning specifically.
The closest prior art is the single-axis problem-first organization
of [Springer NCAA 2026], which addresses distribution shift in
offline RL only.

### C. Axis assignment methodology

Each method covered in this paper is assigned to a *primary axis* —
the weakness whose response motivated the method's introduction — and
zero or more *secondary axes* — additional weaknesses the method
incidentally addresses or partially mitigates. Primary assignment is
made on the basis of the method's stated motivation in its original
publication and the predominant mechanism of its contribution.
Where the original publication's motivation differs from our axis
assignment (notably for distributional methods in §IV.D and Dueling
DQN in §IV.H), the reinterpretation is argued explicitly within the
section.

Cross-references between axis-sections — e.g., Rainbow appearing in
§IV.B (primary) and being cross-referenced from §IV.A, §IV.C, §IV.D,
and §IV.H — surface the connective tissue that catalogue-style
surveys obscure. Appendix A provides a legacy indexer mapping each
method to both the problem-first axis assignment and the prior
method-type taxonomy, supporting readers who arrive expecting the
older organization.

### D. Empirical evidence sources

Section V analyzes Atari benchmark results extracted from the
original papers introducing each method. Section VI reports tabular
benchmarks from controlled re-implementations on Gymnasium
environments (FrozenLake-v1, Taxi-v3, CliffWalking-v1). Section VII
analyzes algorithmic coverage across six widely-used open-source
deep RL repositories (Tianshou [6], XuanCe [7], CleanRL [8], DQN
Zoo [9], Stable Baselines3 [10], RLlib [11]). The three evidence
streams together support different aspects of the paper's
contribution: the literature analysis demonstrates methodological
diversity, the controlled experiments isolate algorithmic from
architectural effects, and the repository analysis maps the
practical landscape of available implementations.

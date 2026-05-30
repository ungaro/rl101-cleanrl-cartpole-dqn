# IV. Related Works {#sec-iv}

Over the past three decades, Q-learning and its deep variants have
evolved through a long sequence of mechanism-level innovations.
Section II.B identified eight foundational weaknesses of vanilla
Q-learning that motivate the field's trajectory; this section
surveys the methods that respond to each. The organization is
problem-first: each subsection corresponds to one weakness,
groups responses by mechanism, and compares them on the trade-offs
they introduce.

**The method-type taxonomy.** Prior Q-learning surveys
[@urtans_2018_pygame; @jang_2019_qsurvey; @boppiniti_2021_evolution; @hafiz_2023_dqnsurvey]
organize methods by *type* — what kind of mechanism a method is. We
call this the *method-type taxonomy* and use that term throughout.
The table below lists its six categories and, crucially, shows where
each category's methods fall among the eight problem axes. The categories
cut *across* the axes: statistical methods split between exploration
(§IV.C) and credit assignment (§IV.D), and ensemble methods between
overestimation control (§IV.A) and exploration (§IV.C). A problem-first
organization regroups exactly these cross-cutting cases by the weakness
each method addresses rather than by what the method is. (The full
per-method mapping is provided as supplementary material.)

| Method-type category | Groups methods that… | Examples | Problem axes |
|---|---|---|---|
| Statistical | model the return distribution or inject parameter noise | C51, QR-DQN, NoisyNet | §IV.C, §IV.D |
| Q-Function Computation | alter how the bootstrap target is computed | Double DQN, Dueling, Rainbow | §IV.A, §IV.B, §IV.H |
| Memory / Replay | change what experience is stored and replayed | DQN, PER, DQfD, HER | §IV.B |
| Ensemble-Based | maintain several Q-estimators | Bootstrapped DQN, EBQL, REDQ | §IV.A, §IV.C |
| Model-Based | use a model or a posterior over values | Bayesian Q, Posterior-Sampling DQN | §IV.C, §V |
| Pure Q-Learning | tabular or minimal value iteration | Q-learning, SARSA, NFQ | §V, §IV.H |

: The method-type taxonomy used by prior Q-learning surveys — its six
categories and where each category's methods fall among this paper's
eight problem axes. Families that postdate this taxonomy (offline RL
§IV.E, multi-agent §IV.F, distributed/meta §IV.G) have no method-type
category at all.

This subsection orients the reader through three further artifacts: a
method genealogy showing the inheritance relationships between
Q-learning's variants, a companion figure showing the modern-RL
branches that have emerged outside the conventional taxonomy, and
an axis × mechanism-family matrix summarizing what mechanism types
are deployed against each weakness. The §IV subsections that
follow each provide a five-part structure: formal statement of
the weakness (subsection A), grouped solution families
(subsection B), trade-offs (subsection C), empirical evidence
(subsection D), open questions (subsection E), and a comparison
summary with positioning grid (subsection F).

### A. Method genealogy

Figure 1 shows the genealogy of Q-learning methods covered in this
paper. Nodes are methods; edges are inheritance relationships
labeled by *the weakness of the parent that the child method
addresses*. The annotations on the edges are themselves a
contribution: they trace the field's evolution as a sequence of
targeted responses rather than a chronological accumulation of
techniques.

```mermaid
%% caption: Method genealogy of Q-learning and its deep variants. Edges are labeled by the weakness of the parent that the child method addresses.
flowchart TD
    Q[Q-Learning<br/>Watkins 1992<br/>tabular ε-greedy single estimator]
    Q -->|overestimation via max| DoubleQ[Double Q-Learning 2010<br/>§IV.A]
    Q -->|on-policy| SARSA[SARSA 1994<br/>§V foundations]
    Q -->|credit assignment| MultiStep[Multi-step Q 1996<br/>§IV.D]
    Q -->|function approximation| NFQ[NFQ 2005<br/>§V foundations]
    NFQ -->|replay + targets| DQN[DQN 2013<br/>replay buffer + CNN<br/>§IV.B]
    DQN -->|target network for stability| NatureDQN[Nature DQN 2015<br/>§IV.H]
    NatureDQN -->|partial observability| DRQN[DRQN 2015<br/>§IV.G]
    NatureDQN -->|deep RL + decoupled| DoubleDQN[Double DQN 2016<br/>§IV.A]
    NatureDQN -->|V/A decomposition| Dueling[Dueling DQN 2016<br/>§IV.H stab/decomp]
    NatureDQN -->|TD-error sampling| PER[Prioritized ER 2016<br/>§IV.B]
    DoubleDQN -->|k-estimator generalization| EBQL[EBQL 2021<br/>§IV.A ensemble bias ctrl]
    EBQL -->|push toward under-est| REDQ[REDQ 2021<br/>§IV.A min-of-M]
    DoubleDQN --> Rainbow[Rainbow 2018<br/>Double + PER + Dueling + n-step<br/>+ Distributional + NoisyNet<br/>§IV.B primary]
    PER --> Rainbow
    Dueling --> Rainbow
    PER -->|demonstrations| DQfD[DQfD 2018<br/>§IV.B]
    PER -->|memory-efficient| MeDQN[MeDQN 2023<br/>§IV.B + §IV.H]
    DQN -->|richer return signal| C51[C51 2017<br/>§IV.D]
    C51 -->|adjustable quantiles| QRDQN[QR-DQN 2018<br/>§IV.D]
    QRDQN -->|continuous τ| IQN[IQN 2018<br/>§IV.D]
    IQN -->|learnable τ| FQF[FQF 2019<br/>§IV.D]
```

Two structural observations emerge from the genealogy. First,
**Rainbow [@hessel_2018_rainbow] is the integration point** for five separate
axis responses (Double DQN, PER, Dueling, multi-step,
distributional, NoisyNet) — it is not a single method but a
deliberate composition across axes. Second, **the distributional
family** (C51 → QR-DQN → IQN → FQF) constitutes an unusually long
single-axis lineage compared to most other branches, reflecting the
field's progressive refinement of distributional Q-learning under
the credit-assignment weakness (W4).

### B. Branches outside the conventional taxonomy

Figure 2 below shows three Q-learning branches that emerged outside
the conventional six-category mechanism taxonomy — families that
respond to weaknesses (distribution shift, multi-agent coordination,
distributed scale and meta-adaptation) that the method-type taxonomy has
no category to hold. These are the methods that motivate §IV.E,
§IV.F, and §IV.G.

```mermaid
%% caption: Q-learning branches outside the conventional six-category mechanism taxonomy — offline RL (§IV.E), multi-agent value decomposition (§IV.F), and distributed / meta-learning (§IV.G).
flowchart TD
    NDQN[@mnih_2015_nature]
    NDQN -->|offline data only NEW §IV.E| BCQ[@fujimoto_2019_bcq]
    BCQ --> CQL[@kumar_2020_cql]
    CQL --> IQL[@kostrikov_2022_iql]
    BCQ -.->|ensemble variant| EDAC[@an_2021_edac]
    NDQN -->|multiple cooperating agents NEW §IV.F| VDN[@sunehag_2017_vdn]
    VDN --> QMIX[@rashid_2018_qmix]
    QMIX --> QPLEX[@wang_2021_qplex]
    QMIX -.->|constraint-free| QTRAN[@son_2019_qtran]
    NDQN -->|massive scale NEW §IV.G| ApeX[@horgan_2018_apex]
    ApeX --> R2D2[R2D2 2019]
    R2D2 --> Agent57[@badia_2020_agent57]
    NDQN -->|meta/fast adapt NEW §IV.G| MAML[MAML-Q / Meta-Q 2017+]
```

The visual absence of these branches from prior Q-learning surveys
[@urtans_2018_pygame; @jang_2019_qsurvey; @boppiniti_2021_evolution; @hafiz_2023_dqnsurvey] is one of the most direct empirical arguments for the
structural pivot of §IV. The method-type taxonomy classifies methods by
mechanism type; these branches require new mechanism categories
(constrained policies, value decomposition, distributed actors,
meta-learning) that did not exist when the original taxonomy was
conventionalized.

### C. Axis × mechanism-family matrix

The matrix below summarizes which mechanism families address which
axis. Filled cells indicate a primary contribution; open cells
($\circ$) indicate an incidental or secondary contribution; blanks
indicate the axis is not addressed by that mechanism family.

| Axis                       | Decoupling      | Ensembles       | Architectural decomposition | Behavior / loss modulation | Distributed / parallel |
|---|---|---|---|---|---|
| **W1 Overestimation** (§IV.A) | DoubleQ, DDQN   | EBQL, REDQ      | Dueling (rel.) $\circ$           | —                          | —                      |
| **W2 Sample inefficiency** (§IV.B) | —              | —               | —                           | PER, DQfD, MeDQN, HER      | Ape-X (replay) $\circ$       |
| **W3 Brittle exploration** (§IV.C) | —              | Bootstrapped, UCB-Q | NoisyNet, ParSpaceNoise | CBDQ, PSDQN, RND, Go-Explore | Agent57 (portfolio)  |
| **W4 Reward sparsity** (§IV.D) | —              | —               | Distributional family       | n-step, $\lambda$-returns          | —                      |
| **W5 Distribution shift** (§IV.E) | BCQ            | EDAC            | —                           | CQL, IQL, BRAC, AWAC       | —                      |
| **W6 Multi-agent coord.** (§IV.F) | —              | —               | VDN, QMIX, QPLEX, QTRAN     | —                          | —                      |
| **W7 Scaling / adaptation** (§IV.G) | —              | —               | DRQN, R2D2                  | —                          | Ape-X, Agent57, Meta-Q |
| **W8 Function-approx stability** (§IV.H) | —      | —               | Target net, Dueling, PQN    | MeDQN consol., Munchausen  | —                      |

Several patterns are visible at a glance:

- The **architectural decomposition** column is the most populous
  — most axes have at least one architectural response. This
  reflects the field's strong preference for *structural* solutions
  (modifying what the network represents) over *training-procedure*
  solutions (modifying what the loss optimizes).
- The **distributed** column is sparse but increasingly populated
  by frontier agents (Ape-X, Agent57). The pattern suggests
  scaling-as-mechanism is undercovered in the literature relative
  to its empirical importance — a point the open-questions
  subsection of §IV.G develops.
- The **decoupling** column has only two entries (DoubleQ-derived).
  Decoupling is mechanistically narrow; ensembling subsumes most
  of its effect once compute budgets permit $K > 2$ Q-functions.
- The **bottom-left quadrant** of the matrix (W6, W7 × decoupling
  or ensembles) is empty. Whether this reflects a true gap in the
  literature or merely a labeling artifact is an open question
  flagged in §VIII.

### D. Section roadmap

The remainder of §IV proceeds axis by axis:

- **§IV.A** Overestimation bias — Double Q-learning to ensemble
  bias control (EBQL, REDQ).
- **§IV.B** Sample inefficiency — prioritized replay, demonstrations,
  memory-efficient consolidation, hindsight relabeling.
- **§IV.C** Brittle exploration — noise injection, ensemble
  disagreement, belief modulation, intrinsic motivation, archival
  exploration.
- **§IV.D** Reward sparsity and credit assignment — multi-step
  returns and the distributional family.
- **§IV.E** Distribution shift (offline RL) — policy constraint,
  value penalty, expectile regression, ensemble diversification.
- **§IV.F** Multi-agent coordination — additive, monotonic,
  duplex-dueling, and constraint-free value decomposition.
- **§IV.G** Scaling and slow adaptation — distributed actor-learner
  architectures, bandit-controlled exploration meta-policies,
  meta-learning, recurrence.
- **§IV.H** Function-approximation instability — target networks,
  architectural decomposition, normalization recipes, consolidation
  losses.
- **§IV.I** Theoretical foundations and recent advances —
  convergence theory, finite-time bounds, pessimism in offline RL,
  stability theory, and IGM theorems. A meta-section that surveys
  the theoretical landscape underlying the eight axes above.
- **§IV.J** Q-learning for foundation model alignment —
  Q-Transformer, ShiQ, VLM Q-Learning, Q$^\sharp$. An application-oriented
  section covering Q-learning's role in fine-tuning large language
  and vision-language models, distinct from the weakness-axes in
  that it represents a new deployment regime rather than a new
  mechanism category.

Each axis subsection (A–H) ends with **F. Comparison summary**: a
uniform six-column table (method / mechanism category / mechanism /
cost / best-at / empirical anchor) and, where the trade-off is
naturally two-dimensional, a 2D positioning grid. §IV.E additionally
provides a method-selection decision tree for practitioner use.
§IV.I substitutes a results-by-year theoretical-status table for the
comparison grid since its entries are theorems rather than methods.
§IV.J provides a four-column comparison table organized by
off-policy support and KL-regularization compatibility.

---
title: "Understanding Q-Learning and Deep Q-Learning in 2025: A Methodological and Empirical Survey"
abstract: |
  Q-learning remains a cornerstone of Reinforcement Learning (RL),
  underpinning many state-of-the-art deep RL algorithms. Yet, despite
  its foundational status, no survey has systematically organized the
  diverse innovations that have shaped Q-learning over the past three
  decades. This paper fills that gap by presenting a comprehensive
  review of Q-learning variants, organized around the eight
  foundational weaknesses of vanilla Q-learning that modern methods
  address — overestimation bias, sample inefficiency, brittle
  exploration, reward sparsity and credit assignment, distribution
  shift, multi-agent coordination, slow adaptation, and
  function-approximation instability. We introduce a unified taxonomy
  covering both tabular and deep Q-learning methods, including offline
  RL, multi-agent value decomposition, and distributed Q-learning, and
  compile benchmark results from original papers across Atari and
  classic control environments. To complement the literature review,
  we provide a comparative analysis of six major open-source deep RL
  repositories — Tianshou, XuanCe, CleanRL, DQN Zoo, Stable
  Baselines3, and RLlib — highlighting their coverage of Q-learning
  algorithms and practical implementation challenges. Together, these
  contributions offer a structured resource for researchers and
  practitioners to navigate the evolution, empirical performance, and
  implementation landscape of Q-learning.
documentclass: article
geometry: margin=1in
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: blue
header-includes: |
  \usepackage{fancyhdr}
  \pagestyle{fancy}
  \fancyhf{}
  \fancyhead[L]{\small\itshape\nouppercase{\leftmark}}
  \fancyhead[R]{\small\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
---

## Impact Statement

Q-learning has influenced nearly every area of deep Reinforcement
Learning, yet its own methodological evolution has lacked a focused
and analytical treatment. This paper addresses that need by
delivering a problem-first taxonomy of Q-learning methods,
benchmarking results from original studies, and a comparative
analysis of leading open-source implementations. By bridging
theoretical advances with empirical evidence and practical
repositories, this work clarifies the algorithmic landscape of
Q-learning and highlights both underexplored areas and reproducibility
challenges. The resulting insights are intended to guide the
development of more robust, efficient, and extensible deep RL systems
that build on Q-learning principles.

**Index Terms:** Deep Learning, Machine Learning, Reinforcement
Learning, Q-Learning, Survey

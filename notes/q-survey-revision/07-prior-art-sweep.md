# Prior-Art Sweep — 2024–2026 Q-Learning Surveys

Findings from a focused web sweep (Google Scholar, arXiv cs.LG / cs.AI,
Semantic Scholar) on whether other surveys in the 2024–2026 window
threaten our differentiation claims.

This file answers the open question parked at the end of
`01-pitch-analysis.md`.

---

## Verdict

The **problem-first reorganization angle is OPEN.** No 2024–2026
Q-learning / DQN survey takes that organizing structure. The closest
partial overlaps are (a) one broad RL survey that is method-family
organized but discusses challenges as a secondary lens, and (b) a
single-axis (distribution shift) offline-RL survey. **No paper unifies
tabular + deep Q-learning into a problem-axis framework.**

The structural pivot in Suggestion A remains a clean differentiation
claim. Two caveats apply (see below) — one nontrivial.

---

## Surveys that materially overlap our scope

### 1. Ghasemi, Moosavi & Ebrahimi (2024/2025)
**"A Comprehensive Survey of Reinforcement Learning: From Algorithms
to Practical Challenges."** arXiv:2411.18892, Nov 2024 / rev Feb 2025.

- **Scope:** broad RL, tabular + deep, value/policy/actor-critic.
- **Organization:** method family (value-based → policy → actor-critic),
  sub-split tabular vs. approximation. *Not* problem-first.
- **Repository comparison:** No.
- **Atari benchmarks:** No (not in visible content).
- **Q-learning coverage:** Q-learning, Double Q, DQN, DDQN, Dueling
  DQN. **PER, Rainbow, distributional DQN apparently absent.**

**This is the closest competitor and the most important paper to
position against.** Our unified tabular+deep treatment with finer
method-type categories is a more granular taxonomy than theirs, and
our repo/Atari analysis is unique. Direct citation + one-paragraph
positioning is needed.

### 2. Springer NCAA (2026) — paywalled, abstract only
**"Distribution shift, generalization and OOD challenge in offline
reinforcement learning: a comprehensive survey."**
DOI: 10.1007/s00521-026-11966-8.

- **Scope:** offline RL only.
- **Organization:** **problem-first along one axis** (distribution
  shift), with four method buckets — Q-value restriction,
  uncertainty-based Q-restriction, policy constraint,
  uncertainty-based policy constraint.

**This is the closest precedent for problem-first organization, but
it covers ONE of our eight axes and only offline RL.** Cite as prior
art for our proposed §IV.E (Distribution Shift). Their existence
actually *strengthens* our positioning: a single-axis problem-first
treatment exists; a multi-axis one does not.

### 3. Hundal, Xiao, Cao, Dong & Rigger (2025) — NOT A SURVEY but threat-adjacent
**"On the Mistaken Assumption of Interchangeable Deep Reinforcement
Learning Implementations."** arXiv:2503.22575.

- Empirically benchmarks Stable-Baselines3, CleanRL, Baselines, RLlib,
  Tianshou on PPO across 56 Atari games.
- Finds SB3/CleanRL/Baselines reach superhuman in ~50% of trials,
  RLlib/Tianshou <15%.
- **This directly overlaps our contribution #3 (repo comparison).**

**Material threat.** Reviewers will absolutely flag this against our
repo comparison if we don't cite it and distinguish. Our analysis is
taxonomic/feature-oriented; theirs is empirical reproducibility audit
on PPO. The angles are complementary, but the distinction needs to be
explicit in our prose. One sentence in §VII is the minimum.

### 4. Murphy (2024/2025)
**"Reinforcement Learning: An Overview."** arXiv:2412.05265, Dec 2024
/ latest Dec 2025.

- **Scope:** textbook-style; value/policy/model-based/multi-agent/LLM+RL.
- **Organization:** method type, with explicit LLM-RL angle.
- **Repository comparison:** No. **Atari:** not emphasized.

More "Murphy's notes" than a peer-reviewed survey, but reviewers may
cite it as canonical coverage. Our finer DQN taxonomy and repo
comparison is straightforward to position against; its existence does
not threaten our angle.

---

## Borderline / niche

| Paper | Year | Why it's not a competitor |
|---|---|---|
| Ter, Adetifa & Udekwe — "Taxonomy and Trends in RL for Robotics and Control Systems" (arXiv:2510.21758) | 2025/26 | Robotics-focused, continuous control (DDPG, TD3, PPO, SAC). Minimal DQN. |
| Cheng, Yu & Xing — "A Survey on Explainable Deep RL" (arXiv:2502.06869) | 2025 | Cross-cut by *explanation level*, different axis entirely |
| Yang et al. — "Advancements in Q-learning meta-heuristic optimization algorithms" (Wiley WIREs DMKD, doi:10.1002/widm.1548) | 2024 | Niche — Q-learning as controller for metaheuristics, not core theory |
| "A Survey for Deep RL Based Network Intrusion Detection" (arXiv:2410.07612) | 2024 | Application domain |
| "RL for Robotics: A Survey of Real-World Successes" (arXiv:2408.03539) | 2024 | Application domain |
| IJCAI 2025 — "Reward Models in Deep RL: A Survey" | 2025 | Different cut |

---

## Honest negatives

- Google Scholar and arXiv returned **no dedicated "DQN survey" or
  "Q-learning survey" paper in 2024–2026** that mirrors Hafiz 2022's
  scope. The closest is Ghasemi 2025 (broader). **This is good news
  for our paper's positioning** — the niche is genuinely empty.
- No 2024–2026 survey organizes **all of RL** by problem/weakness
  axes. Closest partial match is the Springer 2026 offline-RL paper
  on the distribution-shift axis only.
- Despite overestimation / exploration / sample-efficiency being
  recurring problems in the literature, **nobody in 2024–2026 has
  used them as the top-level organizing structure for a
  Q-learning-centered survey.** The problem-first pivot is genuinely
  novel territory.

---

## Implications for the revision plan

1. **The structural pivot in Suggestion A is more defensible than
   `01` claimed.** We have evidence the angle is open. If we go ahead
   with the pivot, we can frame it as "the first multi-axis
   problem-first treatment of Q-learning" in the introduction —
   citing Springer 2026 as the single-axis precedent.

2. **Add three concrete citations to the revised related-works
   framing:**
   - Ghasemi 2024/2025 → cited in §I and §IV as the broad-RL
     comparator we're more granular than
   - Springer 2026 NCAA → cited at the head of §IV.E as the
     single-axis problem-first prior art
   - Hundal 2025 → cited in §VII (repository comparison) as the
     empirical reproducibility-audit counterpart to our taxonomic
     analysis

3. **The repo-comparison contribution needs a clarifying sentence.**
   Something like: "Where Hundal et al. (2025) audit reproducibility
   empirically on PPO, our analysis is taxonomic — we map algorithmic
   coverage across six repositories to identify implementation gaps
   in the Q-learning family." This costs one sentence and inoculates
   the contribution.

4. **Update Table I (the "Comparison of Q-Learning and Deep Q-Learning
   Survey Papers" table).** It currently compares against four
   pre-2023 surveys (Urtans 2018, Jang 2019, Boppiniti 2021, Hafiz
   2022). Add Ghasemi 2024/2025 as a fifth row. The check-mark pattern
   stays favorable to our paper but the table becomes current.

5. **Hold the structural pivot pitch.** The sweep validates it as a
   live differentiator; this is not the moment to retreat to the
   figure-only path (Suggestion B alone).

---

## What I'd still do before submission

- A targeted Semantic Scholar / Connected Papers search seeded by
  Ghasemi 2024/2025 to surface anything that cites or is cited by it
  and might be too recent for general search.
- Check whether any of the recent NeurIPS / ICML 2025 workshop
  proceedings have Q-learning retrospectives — workshops are where
  problem-first framings sometimes appear before they make it to
  full surveys.
- A read of the Springer 2026 paper proper (not just the abstract) —
  the four-bucket problem-axis structure they use is methodologically
  close to what we propose, and a careful read of their framing would
  let us either align with or differentiate from it deliberately.

---

*Sources:*
- arXiv:2411.18892 — Ghasemi et al., Comprehensive Survey of RL
- arXiv:2412.05265 — Murphy, RL: An Overview
- arXiv:2510.21758 — Ter et al., Taxonomy/Trends in RL for Robotics
- arXiv:2502.06869 — Cheng et al., Explainable DRL Survey
- arXiv:2503.22575 — Hundal et al., Interchangeable DRL Implementations
- Springer NCAA 2026, DOI 10.1007/s00521-026-11966-8 — Offline RL Distribution Shift
- Wiley WIREs DMKD 2024, doi:10.1002/widm.1548 — Yang et al., Q-learning Metaheuristics

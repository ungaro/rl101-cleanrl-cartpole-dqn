# Submission Readiness — IEEE TAI

**Synced to v0.24 (2026-05-30).** Status: **submission-ready** pending
the three portal-side items below.

## Package manifest

| Artifact | Location | Status |
|---|---|---|
| Main paper (anonymous, two-column) | `draft-tai/main.pdf` | 19 pp ✅ (≤21 cap) |
| Supplementary material | `draft-tai/supplement.pdf` | 10 pp ✅ |
| Reproduction script | `scripts/tabular_experiments.py` | ✅ committed |
| Experiment results | `draft-tai/data/tabular_results.json` | ✅ committed |
| Extended version (source) | `draft-monograph/` (tag `monograph-v0.15`) | ✅ frozen |
| Build | `build-tai.sh`, `build-supp.sh`, `tables-twocol.lua` | ✅ |

## Automated hygiene checks — all PASS

| Check | TAI requirement | Result |
|---|---|---|
| Anonymization | `\anontrue` on submission build | ✅ main.tex + supplement hardcoded anonymous |
| Author/identity leakage | none in PDF | ✅ ("Wang" = cited authors; no author block printed) |
| Internal-notes leakage | none | ✅ no monograph/changelog/ledger/prior-art strings in either PDF |
| External links | no author-identifying links | ✅ only two cited third-party tool URLs (DeepMind dqn_zoo, PyGame-LE) — legitimate references |
| Title | ≤15 words | ✅ 12 |
| Abstract | ≤250 words, 1 paragraph, no equations | ✅ 179 words, 0 equations |
| Impact statement | 100–150 words | ✅ 135 |
| Keywords | 3–6 | ✅ 5 present |
| Format | two-column IEEEtran PDF | ✅ both documents |
| Acknowledgements / funding | excluded under anonymity | ✅ none present |
| Conclusion + future work | required | ✅ §VIII |

## Portal-side items (cannot be done from the repo)

1. **Keywords from TAI's dropdown.** Our five (Deep Learning, Machine
   Learning, Reinforcement Learning, Q-Learning, Survey) are standard
   IEEE taxonomy terms; select the closest matches from the actual
   submission-portal dropdown.
2. **Similarity / iThenticate check (≤20% or auto-reject).** Run through
   an institutional account before submitting. Risk is low — original
   prose, short cited quotes, and TAI excludes the bibliography and
   properly-quoted text from the count.
3. **ORCID for all nine authors** — required by the portal.

## Reminders

- Submission build uses `\anontrue`. Flip to `\anonfalse` **only** for
  the camera-ready after acceptance (restores the named author block).
- IEEE AI-generated-text disclosure was intentionally not added per
  author decision (the manuscript is a multi-author revision of prior
  work, not from-scratch generation); revisit if policy requires.
- Manuscript category at submission: **Original Research Review
  Manuscript** (review-paper page limits apply).

## Distillation arc (for the record)

47pp monograph → 34 (§IV.A–H) → 26 (§IV.I/J + appendices → supp) → 27
(cross-axis table) → 21 (§V/§VII/§VI) → **19** (front matter). Main +
supplement together hold all the comprehensive content; nothing was
lost, only relocated.

#!/usr/bin/env bash
# Build the IEEE TAI two-column PDF from the section markdown files.
#
# Pipeline:
#   1. render-mermaid.mjs pre-renders supported diagrams to PNGs (zinc-light).
#   2. pandoc converts the concatenated markdown to a LaTeX *fragment*
#      (body.tex) with --natbib, so [@key] -> \cite for IEEEtran.bst.
#   3. main.tex (IEEEtran, journal, two-column) \input{body.tex} and is
#      compiled with the SYSTEM xelatex + bibtex (which see IEEEtran.{cls,bst});
#      local copies in this folder are committed fallbacks.
#
# Usage:
#   bash build-tai.sh            # writes main.pdf into this folder
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Pin SYSTEM TeX engines: conda's bibtex/kpsewhich do NOT see system IEEEtran.bst.
XELATEX=/usr/bin/xelatex
BIBTEX=/usr/bin/bibtex
# Append (not prepend) /usr/bin so the user's nvm/conda node still shadows the
# old system node for mermaid-filter.
export PATH="$PATH:/usr/bin"

# Body order: I..VIII, then appendices A,B. (Bibliography handled by main.tex.)
FILES=(
  1-introduction.md 2-background.md 3-methodology.md
  4-overview.md
  4a-overestimation-bias.md 4b-sample-inefficiency.md 4c-brittle-exploration.md
  4d-reward-sparsity.md 4e-distribution-shift.md 4f-multi-agent.md
  4g-scaling-adaptation.md 4h-stability.md 4i-theoretical-advances.md
  4j-foundation-model-alignment.md
  5-atari-benchmarks.md 6-tabular-empirical.md 7-repositories.md 8-conclusion.md
  A-legacy-indexer.md B-notation-and-proofs.md
)

TMP="$(mktemp -t q-tai-XXXXXX.md)"
trap 'rm -f "$TMP"' EXIT

first=1
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "WARN: missing $f" >&2; continue; }
  if [[ $first -eq 0 ]]; then printf '\n\n' >> "$TMP"; fi
  node ./render-mermaid.mjs "$f" >> "$TMP"
  first=0
done

# §-cross-reference linkification (same map as the monograph build).
perl -i -pe '
  BEGIN {
    %m = (
      "§IV.A"=>"[§IV.A](#sec-iv-a)","§IV.B"=>"[§IV.B](#sec-iv-b)",
      "§IV.C"=>"[§IV.C](#sec-iv-c)","§IV.D"=>"[§IV.D](#sec-iv-d)",
      "§IV.E"=>"[§IV.E](#sec-iv-e)","§IV.F"=>"[§IV.F](#sec-iv-f)",
      "§IV.G"=>"[§IV.G](#sec-iv-g)","§IV.H"=>"[§IV.H](#sec-iv-h)",
      "§IV.I"=>"[§IV.I](#sec-iv-i)","§IV.J"=>"[§IV.J](#sec-iv-j)",
      "§II.B"=>"[§II.B](#sec-ii)","§II.A"=>"[§II.A](#sec-ii)","§II.C"=>"[§II.C](#sec-ii)",
      "§VIII"=>"[§VIII](#sec-viii)","§VII"=>"[§VII](#sec-vii)","§VI"=>"[§VI](#sec-vi)",
      "§V"=>"[§V](#sec-v)","§IV"=>"[§IV](#sec-iv)","§III"=>"[§III](#sec-iii)",
      "§II"=>"[§II](#sec-ii)","§I"=>"[§I](#sec-i)",
    );
    $rx = join "|", map { quotemeta } sort { length($b) <=> length($a) } keys %m;
  }
  s/($rx)/$m{$1}/g;
' "$TMP"

# Markdown -> LaTeX fragment. --natbib turns [@key] into \cite for IEEEtran.bst.
# mermaid-filter catches block types render-mermaid.mjs leaves (quadrantChart).
pandoc "$TMP" \
  -f markdown \
  -t latex \
  --natbib \
  --lua-filter ./tables-twocol.lua \
  --filter mermaid-filter \
  -o body.tex

# Compile: xelatex -> bibtex -> xelatex x2 (resolve refs + cross-refs).
"$XELATEX" -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 || \
  "$XELATEX" -interaction=nonstopmode main.tex   # rerun verbose on failure to surface error
"$BIBTEX" main || true
"$XELATEX" -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
"$XELATEX" -interaction=nonstopmode main.tex >/dev/null 2>&1 || true

echo "Wrote $HERE/main.pdf"

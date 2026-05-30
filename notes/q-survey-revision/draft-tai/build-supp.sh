#!/usr/bin/env bash
# Build the Supplementary Material PDF (separate from the main paper).
# Same pipeline as build-tai.sh: pandoc -> LaTeX fragment (supp-body.tex)
# with --natbib + tables-twocol.lua, compiled with system xelatex + bibtex.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

XELATEX=/usr/bin/xelatex
BIBTEX=/usr/bin/bibtex
export PATH="$PATH:/usr/bin"

FILES=(
  supp-S1-search-log.md
  supp-S2-method-type-index.md
  supp-S3-atari-tables.md
  supp-S4-repository-matrix.md
  supp-S5-notation-proofs.md
  supp-S6-experiment-details.md
)

TMP="$(mktemp -t q-supp-XXXXXX.md)"
trap 'rm -f "$TMP"' EXIT

first=1
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "WARN: missing $f" >&2; continue; }
  if [[ $first -eq 0 ]]; then printf '\n\n' >> "$TMP"; fi
  cat "$f" >> "$TMP"   # no mermaid in the supplement; plain concat
  first=0
done

pandoc "$TMP" \
  -f markdown \
  -t latex \
  --natbib \
  --lua-filter ./tables-twocol.lua \
  -o supp-body.tex

# main file is supplement.tex -> supplement.pdf
"$XELATEX" -interaction=nonstopmode -jobname=supplement supplement.tex >/dev/null 2>&1 || \
  "$XELATEX" -interaction=nonstopmode -jobname=supplement supplement.tex
"$BIBTEX" supplement || true
"$XELATEX" -interaction=nonstopmode -jobname=supplement supplement.tex >/dev/null 2>&1 || true
"$XELATEX" -interaction=nonstopmode -jobname=supplement supplement.tex >/dev/null 2>&1 || true

echo "Wrote $HERE/supplement.pdf"

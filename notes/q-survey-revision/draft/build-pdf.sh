#!/usr/bin/env bash
# Build a single PDF from the section markdown files in this folder.
#
# Requirements (one-time install on Ubuntu/WSL):
#   sudo apt install -y pandoc texlive-xetex texlive-fonts-recommended \
#                       texlive-latex-extra
#   npm install -g beautiful-mermaid sharp mermaid-filter
#
# Pipeline:
#   1. render-mermaid.mjs pre-processes each section, rendering supported
#      diagrams (flowchart, state, sequence, class, ER, xychart-beta) with
#      beautiful-mermaid in the zinc-light theme → PNGs cached in
#      ./mermaid-beautiful/.
#   2. Pandoc + xelatex compiles the concatenated, pre-processed markdown.
#      mermaid-filter still runs as a pandoc filter to catch unsupported
#      block types (notably quadrantChart).
#
# The filenames sort lexically into the paper order:
#   1-introduction → 2-background → 3-methodology → 4-overview →
#   4a..4h axis-sections → 5..8 → A-legacy-indexer
#
# Usage:
#   bash build-pdf.sh                # writes paper.pdf into this folder
#   bash build-pdf.sh out.pdf        # writes to a custom path

set -euo pipefail

OUT="${1:-paper.pdf}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Use system pandoc + xelatex from apt; conda's are often broken
# (mktexlsr.pl missing, xelatex shipped as a shell wrapper).
# We APPEND /usr/bin (not prepend) — otherwise /usr/bin/node (often a
# very old system Node from apt) shadows the user's nvm/conda Node and
# mermaid-filter falls back to ES syntax that old Node can't parse.
if [ -x /usr/bin/pandoc ] && [ -x /usr/bin/xelatex ]; then
  export PATH="$PATH:/usr/bin"
fi

# Concatenate the section files into one markdown source.
# Each section is first pre-processed by render-mermaid.mjs so the
# supported diagram blocks become PNG image references in the zinc-light
# theme. Quadrant charts are left for mermaid-filter to handle.
# Insert a pagebreak between top-level sections so each starts fresh.
TMP="$(mktemp -t q-survey-XXXXXX.md)"
trap 'rm -f "$TMP"' EXIT

first=1
for f in *.md; do
  if [[ "$f" == "$(basename "$OUT" .pdf).md" ]]; then continue; fi
  if [[ $first -eq 0 ]]; then
    printf '\n\\newpage\n\n' >> "$TMP"
  fi
  node ./render-mermaid.mjs "$f" >> "$TMP"
  first=0
done

# Pandoc render with xelatex (unicode-friendly, handles §, ε, ∇, etc.).
# --filter mermaid-filter rasterizes ```mermaid blocks to images.
# --toc inserts a table of contents at the front.
pandoc "$TMP" \
  -o "$OUT" \
  --pdf-engine=xelatex \
  --filter mermaid-filter \
  --toc \
  --toc-depth=2 \
  -V geometry:margin=1in \
  -V documentclass=article \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -V colorlinks=true \
  -V linkcolor=blue \
  --metadata title="Understanding Q-Learning and Deep Q-Learning"

echo "Wrote $OUT"

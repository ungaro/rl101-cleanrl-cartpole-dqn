-- Convert pandoc's default `longtable` output into IEEE `table*` floats.
-- longtable is illegal in two-column mode; `table*` spans both columns and is
-- the correct IEEE construct for wide tables. We render each Table to LaTeX,
-- strip the longtable head/foot machinery, and wrap the result in table*.
--
-- Column widths: pandoc sizes p{} columns relative to \linewidth. Inside a
-- table* that equals the full text width, so wide tables scale sensibly.

function Table(tbl)
  -- Render just this table through pandoc's own LaTeX writer.
  local doc = pandoc.Pandoc({ tbl })
  local latex = pandoc.write(doc, 'latex')

  -- Drop the continuation-header block (between \endfirsthead and \endhead);
  -- it duplicates the header row, which a plain tabular must not repeat.
  latex = latex:gsub('\\endfirsthead.-\\endhead', '')

  -- The longtable foot puts \bottomrule before the body via \endlastfoot.
  -- Remove that misplaced rule+marker; we re-add \bottomrule at the true end.
  latex = latex:gsub('\\bottomrule%s*\\noalign{}%s*\\endlastfoot', '')
  latex = latex:gsub('\\bottomrule%s*\\endlastfoot', '')
  latex = latex:gsub('\\endlastfoot', '')
  latex = latex:gsub('\\endhead', '')
  latex = latex:gsub('\\endfoot', '')

  -- longtable captions sit inside the environment via \tabularnewline; strip
  -- them (we re-attach a clean \caption on the float if one exists).
  local caption = nil
  latex = latex:gsub('\\caption%[(.-)%]{(.-)}\\tabularnewline', function(_, c) caption = c; return '' end)
  latex = latex:gsub('\\caption{(.-)}\\tabularnewline', function(c) caption = c; return '' end)

  -- Remove remaining \noalign{} no-ops left by booktabs-in-longtable.
  latex = latex:gsub('\\noalign{}', '')

  -- Swap the environment: longtable -> tabular, closing with a \bottomrule.
  latex = latex:gsub('\\begin{longtable}%[%]', '\\begin{tabular}')
  latex = latex:gsub('\\begin{longtable}', '\\begin{tabular}')
  latex = latex:gsub('\\end{longtable}', '\\bottomrule\n\\end{tabular}')

  -- Size heuristic: wide tables span both columns (table*); narrow ones stay
  -- in a single column (table). A table is "wide" if it has many columns or a
  -- declared total width that won't fit one ~3.5in column.
  local ncols = #tbl.colspecs
  local wide = ncols >= 4

  local capline = caption and ('\\caption{' .. caption .. '}\n') or ''
  local env = wide and 'table*' or 'table'
  local wrapped = '\\begin{' .. env .. '}[t]\n\\centering\n\\footnotesize\n'
    .. capline .. latex .. '\n\\end{' .. env .. '}'
  return pandoc.RawBlock('latex', wrapped)
end

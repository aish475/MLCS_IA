#!/bin/bash
# Generate tables/figures from results, then compile all three Aishwarya PDFs.
set -e
ROOT="/Users/aditya/honors mlcs/Aishwarya_Gawali_16014223007"
cd "$ROOT"
. "/Users/aditya/honors mlcs/.venv/bin/activate"
python IA2/make_tables_ia2.py
# share generated fragments with IA-1 report and the paper
cp IA2/report/tables/*.tex IA1/report/tables/
cp IA2/report/figs/*.pdf   IA1/report/figs/   2>/dev/null || true
cp IA2/report/tables/*.tex IA2/paper/tables/
cp IA2/report/figs/*.pdf   IA2/paper/figs/
build () {  # dir base
  cd "$ROOT/$1"
  pdflatex -interaction=nonstopmode "$2".tex >/dev/null 2>&1
  bibtex "$2" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$2".tex >/dev/null 2>&1
  pdflatex -interaction=nonstopmode "$2".tex > "$2".build.log 2>&1
  local err=$(grep -c "^!" "$2".build.log || true)
  local pg=$(pdfinfo "$2".pdf 2>/dev/null | awk '/Pages/{print $2}')
  echo "$2: errors=$err pages=$pg"
  cp "$2".pdf "$3".pdf
  rm -f *.aux *.log *.out *.bbl *.blg *.toc "$2".build.log
}
build IA1/report IA1_Aishwarya_Gawali IA1_Aishwarya_Gawali_16014223007
build IA2/report IA2_Aishwarya_Gawali IA2_Aishwarya_Gawali_16014223007
build IA2/paper  paper_Aishwarya      paper_Aishwarya_Gawali_16014223007
echo "ALL COMPILED"

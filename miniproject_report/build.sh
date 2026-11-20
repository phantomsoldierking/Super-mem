#!/bin/sh
# install packages missing from the medium scheme, then compile with bibtex
tlmgr install easy xifthen ifmtarg tocbibind ntgclass nicematrix tablefootnote threeparttable xpatch sttools multirow adjustbox natbib algorithms >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main > bibtex.out 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1

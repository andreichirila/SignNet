#!/bin/sh

pdflatex main.tex
bibtex main.aux
bibtex web.aux
makeindex main
pdflatex main.tex
pdflatex main.tex
open main.pdf&
 

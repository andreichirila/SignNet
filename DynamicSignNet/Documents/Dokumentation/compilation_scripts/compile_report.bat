pdflatex main.tex
bibtex main.aux
bibtex web.aux
makeindex -s softeng.ist main.idx
pdflatex main.tex
pdflatex main.tex
pop -f main.pdf
 

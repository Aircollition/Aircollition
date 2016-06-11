# makefile for beamer
# Install rubber with homebrew:
# brew install homebrew/tex/rubber
# By default beamer has no reference
# it has to be made by hand so no bibtex
FILE = Soutenance
TEX = pdflatex

all: pdf clean view

pdf: $(FILE).tex 
	rubber -m pdftex $<
	
edit: 
	mate $(FILE).tex

view: $(FILE).pdf
	open $(FILE).pdf

clean:
	rm -f *.log *.aux *.nav *.out *.snm *.toc

cleanall:
	rm -f *.log *.aux *.nav *.out *.snm *.toc $(FILE).pdf
	
correct: $(FILE).txt
	aspell --lang=fr --mode=tex check $(FILE).tex

text: $(FILE).tex
	detex -t $(FILE).tex > $(FILE).txt

.PHONY: all pdf edit view clean cleanall
.SILENT: pdf
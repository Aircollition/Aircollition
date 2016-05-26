# makefile for pdflatex
# Install rubber with homebrew:
# brew install homebrew/tex/rubber
FILE = rapport_Modal_SNA
TEX = pdflatex
BIB = bibtex

all: pdf clean view

pdf: $(FILE).tex 
	rubber -m pdftex $<
	
edit: 
	mate $(FILE).tex

view: 
	open $(FILE).pdf

clean:
	rm -f *.aux *.blg *.out *.bbl *.log *.dvi *.toc

cleanall:
	rm -f *.aux *.blg *.out *.bbl *.log *.dvi *.toc $(FILE).pdf

.PHONY: all pdf edit view clean cleanall
.SILENT: pdf
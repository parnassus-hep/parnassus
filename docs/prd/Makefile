# Makefile for Parnassus PRD document
# Usage: make          (builds prd.pdf)
#        make clean    (removes build artifacts)
#        make watch    (re-builds on changes, requires latexmk)

MAIN = prd
TEX  = pdflatex
BIB  = bibtex

.PHONY: all clean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex figures/*.pdf
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

watch:
	latexmk -pdf -pvc $(MAIN)

clean:
	rm -f $(MAIN).{aux,log,out,toc,lof,lot,bbl,blg,nav,snm,vrb,fls,fdb_latexmk,synctex.gz} $(MAIN).pdf

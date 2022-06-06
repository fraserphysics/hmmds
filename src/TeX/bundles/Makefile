# Makefile: For for the single file bundles.pdf

# For pointers on style, see:
# https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project
# and
# http://mirrors.ibiblio.org/CTAN/support/latex-make/latex-make.pdf

ROOT = ../..
XFIGS = $(ROOT)/plotscripts/xfigs
ApneaPlotScripts = $(ROOT)/plotscripts/apnea

# Default target is bundles.pdf
bundles.pdf:

include $(XFIGS)/Rules.mk # Defines $(INTRODUCTION_FIGS) and $(BASIC_ALGORITHMS_FIGS)
include $(ApneaPlotScripts)/Rules.mk # Defines $(APNEA_FIGS)

## bundles.pdf                    : Document with my thoughts on decoding class
bundles.pdf: bundles.tex $(INTRODUCTION_XFIGS) $(BASIC_ALGORITHMS_XFIGS) $(APNEA_FIGS)
	pdflatex bundles
	pdflatex bundles

## clean                          : Remove derived files in this directory
.PHONY : clean
clean:
	rm -f *.aux *.fls *.fdb_latexmk *.pdf *.log

## help                           : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Makefile

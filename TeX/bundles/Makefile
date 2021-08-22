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

include $(XFIGS)/Rules.mk
include $(ApneaPlotScripts)/Rules.mk

bundles.pdf: bundles.tex $(INTRODUCTION_FIGS) $(BASIC_ALGORITHMS_FIGS) $(APNEA_FIGS)
	pdflatex bundles
	pdflatex bundles

.PHONY : clean
clean:
	rm -f *.aux *.fls *.fdb_latexmk *.pdf *.log

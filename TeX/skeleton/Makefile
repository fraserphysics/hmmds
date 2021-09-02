# Makefile: For for the single file figures.pdf

# For pointers on style, see:
# https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project
# and
# http://mirrors.ibiblio.org/CTAN/support/latex-make/latex-make.pdf

ROOT = ../..
PLOTSCRIPTS = $(ROOT)/plotscripts
XFIGS = $(PLOTSCRIPTS)/xfigs
IntroductionPlotScripts = $(PLOTSCRIPTS)/introduction

ApneaPlotScripts = $(PLOTSCRIPTS)/apnea

# Default target is figures.pdf
figures.pdf:

include $(XFIGS)/Rules.mk # Defines $(INTRODUCTION_XFIGS) and $(BASIC_ALGORITHMS_FIGS)
include $(IntroductionPlotScripts)/Rules.mk

include $(ApneaPlotScripts)/Rules.mk # Defines $(APNEA_FIGS)

## figures.pdf                   : Comments on software and all figures for book
figures.pdf: figures.tex $(INTRODUCTION_XFIGS) $(BASIC_ALGORITHMS_XFIGS) $(APNEA_FIGS)
	export TEXINPUTS=$(abspath $(ROOT))//:; pdflatex figures.tex; pdflatex figures.tex

## clean                         : Remove derived files in this directory
.PHONY : clean
clean:
	rm -f *.aux *.pdf *.log

## help                          : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Makefile

# Makefile: For building simple targets locallay

# For pointers on style, see:
# https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project
# and
# http://mirrors.ibiblio.org/CTAN/support/latex-make/latex-make.pdf

ROOT = ..
XFIGS = $(ROOT)/plotscripts/xfigs
ApneaPlotScripts = $(ROOT)/plotscripts/apnea

# Default target is bundles.pdf
bundles.pdf:

include $(XFIGS)/Rules.mk
include $(ApneaPlotScripts)/Rules.mk

bundles.pdf: bundles.tex $(INTRODUCTION_FIGS) $(BASIC_ALGORITHMS_FIGS) $(APNEA_FIGS)
	pdflatex bundles
	pdflatex bundles

# ds21 was derived from ii in ~/projects/hmmds3/TeX/
includes21/%.pdf : includes21/%.fig
	fig2dev -L pdftex -F $< $@
includes21/%.pdf_t: includes21/%.fig includes21/%.pdf
	fig2dev -L pdftex_t -p $(abspath includes21/$*.pdf) $< $@

DS21FIGS = $(call ADD_PDF_PDF_T, includes21, happy apnea_w normal_0)

# The // after TeX includes subdirectories.  The : after // includes
# the standard latex search path
ds21.pdf: ds21.tex $(DS21FIGS)
	export TEXINPUTS=/mnt/precious/home/andy_nix/projects/dshmm/TeX//:; pdflatex ds21.tex; pdflatex ds21.tex

.PHONY : variables
variables:
	@echo $(DS21FIGS)

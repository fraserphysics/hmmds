# Makefile: For SIAM conference on Dynamical Systems 2021

# For pointers on style, see:
# https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project
# and
# http://mirrors.ibiblio.org/CTAN/support/latex-make/latex-make.pdf

ROOT = ../../..
XFIGS = $(ROOT)/src/plotscripts/xfigs
ApneaPlotScripts = $(ROOT)/src/plotscripts/apnea
ABSROOT = $(abspath $(ROOT))

## slides.pdf                     : Slides for my talk
slides.pdf:

include $(XFIGS)/Rules.mk

# ds21 was derived from ii in ~/projects/hmmds3/TeX/
includes/%.pdf : includes/%.fig
	fig2dev -L pdftex -F $< $@
includes/%.pdf_t: includes/%.fig includes/%.pdf
	fig2dev -L pdftex_t -p $(abspath includes/$*.pdf) $< $@

# ADD_PDF_PDF_T is from $(XFIGS)/Rules.mk.  The following dependencies
# should be included but are built elsewhere: book_cover.jpg error.jpg
# a03erN.pdf a03erA.pdf class_net.pdf class_net.pdf_t
DS21FIGS = $(call ADD_PDF_PDF_T, includes, happy apnea_w normal_0)

# 
slides.pdf: slides.tex $(DS21FIGS)
# The // after TeX includes subdirectories.  The : after // includes
# the standard latex search path
	export TEXINPUTS=$(ABSROOT)//:; pdflatex slides.tex; pdflatex slides.tex

## words.ps                       : The words to say for my talk
words.ps: slides.tex
	tail +250 slides.tex |enscript --output=words.ps --font=Times-Roman16

## clean                          : Remove derived files
.PHONY : clean
clean:
	rm -f *.aux *.out *nav *.toc *.snm *.pdf *.log *.pdf *.ps includes/*pdf*

.PHONY : variables
variables:
	@echo DS21FIGS  :  $(DS21FIGS)
	@echo ABSROOT   :  $(ABSROOT)
	@echo includes  :  $(abspath includes)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

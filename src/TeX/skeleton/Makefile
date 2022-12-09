# Makefile: For for the single file figures.pdf

# Because this does not import other Rules.mk files, it is not aware
# of the pdfs that figures.tex needs, and it can't build them.
# $(ROOT)/Makefile does import those other Rules.mk files.

# For pointers on style, see:
# https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project
# and
# http://mirrors.ibiblio.org/CTAN/support/latex-make/latex-make.pdf

ROOT = ../../..
BUILD = $(ROOT)/build
TEX = ../

##                               : Default target is figures.pdf in build dir
$(BUILD)/TeX/skeleton/figures.pdf:

include Rules.mk

## clean                         : Remove files made by pdflatex et al.
.PHONY : clean
clean:
	rm -rf $(BUILD)/TeX/skeleton

## help                          : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Makefile

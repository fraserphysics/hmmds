# Makefile: For testing scripts in this directory from this directory.

ROOT = ../..
FIGS_INTRO = $(ROOT)/figs/introduction
IntroPlotScripts = ./

# The default target is test
test:

# Rules.mk contains the real targets and rules to build them
include Rules.mk

## test          : Build all targets
.PHONY : test
test: $(ALL)

## list          : List all files built by this makefile
.PHONY : list
list:
	ls -l $(ALL)

## clean         : Remove all files built by this makefile
.PHONY : clean
clean:
	rm -f $(ALL) *pdf*

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo ALL: $(ALL)

## help          : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Rules.mk
	@sed -n 's/^## / /p' Makefile

## yapf          : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

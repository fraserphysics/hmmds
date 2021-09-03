# Makefile: For testing scripts in this directory from this directory.

ROOT = ../..
BUILD = $(ROOT)/build
FIGS_INTRO = $(BUILD)/figs/introduction
IntroPlotScripts = ./

## ALL           : All of the targets that Rules.mk is responsible for
ALL = $(FIGS_INTRO)/TSintro.pdf

## test          : Build all targets; the default target.
.PHONY : test
test: $(ALL)

# Rules.mk contains the real targets and rules to build them
include Rules.mk

## list          : List all files built by this makefile
.PHONY : list
list:
	ls -l $(ALL)

## clean         : Remove all files built by this makefile
.PHONY : clean
clean:
	rm -f $(ALL)

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

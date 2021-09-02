# Makefile: For testing scripts in this directory from this directory

ROOT = ../..
SYNTHETIC_DATA = $(ROOT)/derived_data/synthetic
SYNTHETIC_CODE = ./
#$(ROOT)/hmmds/synthetic

# The default target is test
test:

# Rules.mk contains the real targets and rules to build them
include Rules.mk

# ALL is all the targets that Rules.mk is responsible for
ALL = $(SYNTHETIC_DATA)/lorenz.flag

## test         : Build all targets
.PHONY : test
test: $(ALL)

## list         : List all files built by this makefile
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
	@echo BASIC_ALGORITHMS_FIGS: $(BASIC_ALGORITHMS_FIGS)

## help                           : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Makefile

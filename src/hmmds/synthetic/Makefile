# Makefile: For testing scripts in this directory from this directory

ROOT = ../../..
BUILD = $(ROOT)/build
HMMDS = $(ROOT)/src/hmmds
N_TRAIN = 50

# The default target is test
test:

# Rules.mk contains the real targets and rules to build them
include Rules.mk

# ALL is all the targets that Rules.mk is responsible for
ALL = $(addprefix $(SYNTHETIC_DATA)/, \
lorenz.flag states m12s.4y lorenz.xyz TrainChar SGO vstates)

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
	@echo SYNTHETIC_CODE: $(SYNTHETIC_CODE)
	@echo SYNTHETIC_DATA: $(SYNTHETIC_DATA)

## help                           : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Makefile

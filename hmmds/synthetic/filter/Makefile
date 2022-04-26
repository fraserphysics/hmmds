# Makefile: For testing scripts in this directory from this directory

ROOT = ../../..
BUILD = $(ROOT)/build

# The default target is test
test:

# Rules.mk contains the real targets and rules to build them
include Rules.mk

# ALL is all the targets that Rules.mk is responsible for
ALL = $(addprefix $(FILTER_DATA)/, \
distribution_data \
linear_map_data \
linear_sde_data \
log_likelihood_data \
linear_particle_data \
lorenz_data \
lorenz_particle_data \
)

## test                 : Build all targets
.PHONY : test
test: $(ALL)

## lorenz_sde_library   : Compile lorenz_sde.pyx
.PHONY : lorenz_sde_library
lorenz_sde_library : lorenz_sde.cpython-38-x86_64-linux-gnu.so

## comparison           : Exercise hmm.statespace.NonLinear
.PHONY : comparison
comparison: compare.py $(FILTER_DATA)/mimic_data $(FILTER_DATA)/linear_map_data
	python $^

## list                 : List all files built by this makefile
.PHONY : list
list:
	ls -l $(ALL)

## clean                : Remove all files built by this makefile
.PHONY : clean
clean:
	rm -f $(ALL) *pdf*

## variables            : Print selected variables.
.PHONY : variables
variables:
	@echo FILTER_DATA: $(SYNTHETIC_CODE)

## help                 : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Makefile

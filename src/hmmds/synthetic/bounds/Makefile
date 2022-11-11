ALL_FIGS =  ToyTS1.pdf ToyStretch.pdf ToyH.pdf benettin.pdf LikeLor.pdf

# Makefile: For testing scripts in this directory from this directory

ROOT = ../../../..
BUILD = $(ROOT)/build

# The default target is test
test:

# Rules.mk contains the real targets and rules to build them
include Rules.mk

# ALL is all the targets that Rules.mk is responsible for
ALL = $(addprefix $(BOUNDS_DATA)/, data_h_view toy_h.data BenettinData like_lor_data)

## test                 : Build all targets
.PHONY : test
test: $(ALL)

data_h_view: h_view.py
	@echo To make data_h_view, run \"python h_view.py\" and press the \"save\" button

toy_h.data: toy_h.py
	python $< --n_t 200 --t_steps .02 .53 .125 --log_steps -3.5 -5.6 -.5 $@

BenettinData: benettin_data.py
	python benettin_data.py --n_times=400 --n_runs=100 --n_relax 50 $@

# This takes almost 8 minutes on cathcart
like_lor_data: like_lor.py
	python $< $@


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

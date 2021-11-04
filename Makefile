# Master Makefile for dshmm

# This includes files named Rules.mk from subdirectories.  Those files
# use the variables ROOT and BUILD defined here to define source and
# destination paths.  That means that I must modify affected Rules.mk
# files if I change the directory structure.

# If subdirectories contain local Makefiles, they are for testing
# local code.  Those Makefiles are not called or used elsewhere.

N_TRAIN = 50

# Look at: https://makefiletutorial.com/

ROOT = .
#ROOT = $(abspath ./)
BUILD = $(ROOT)/build
XFIGS = $(ROOT)/plotscripts/xfigs
ApneaPlotScripts = $(ROOT)/plotscripts/apnea

# Default target
## skeleton                       : Explanation of how I make each figure for the book
.PHONY : skeleton
skeleton: $(BUILD)/TeX/skeleton/figures.pdf

# Rules for making plots
include $(ROOT)/plotscripts/introduction/Rules.mk
include $(ROOT)/plotscripts/basic_algorithms/Rules.mk
include $(ROOT)/plotscripts/variants/Rules.mk
include $(XFIGS)/Rules.mk
include $(ApneaPlotScripts)/Rules.mk

# Rules for making data files
include $(ROOT)/hmmds/synthetic/Rules.mk

# Rules for making documents
include $(ROOT)/TeX/skeleton/Rules.mk

## ds21.pdf                       : Slides for 2021 SIAM Dynamical Systems meeting
.PHONY : ds21.pdf
ds21.pdf : TeX/ds21/slides.pdf

TeX/ds21/slides.pdf:
	cd TeX/ds21 && $(MAKE) slides.pdf

TeX/bundles.pdf: TeX/bundles.tex  $(INTRODUCTION_FIGS) $(BASIC_ALGORITHMS_FIGS) $(APNEA_FIGS)
	cd TeX && $(MAKE) bundles.pdf

#ToDo: Ensure that derived_data/apnea/pass1_report.pickle is up to
#date using hmmds/applications/apnea/Rules.mk
figs/pass1.pdf: plotscripts/apnea/pass1.py derived_data/apnea/pass1_report.pickle
	python $^ $@

derived_data/apnea/pass1_report.pickle:
	cd hmmds/applications/apnea && $(MAKE) pass1_report

.PHONY : test
test:
	ls $(APNEA_FIGS)

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" hmmds/ plotscripts/

## check-types                    : Checks type hints
.PHONY : check-types
check-types:
	export MYPYPATH=$$PYTHONPATH; mypy --no-strict-optional hmmds
# --no-strict-optional allows None as default value

## lint                           : Run pylint
.PHONY : lint
lint :
	pylint --rcfile=pylintrc hmmds plotscripts

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo INTRODUCTION_FIGS: $(INTRODUCTION_FIGS)
	@echo BASIC_ALGORITHMS_FIGS: $(BASIC_ALGORITHMS_FIGS)
	@echo APNEA_FIGS: $(APNEA_FIGS)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

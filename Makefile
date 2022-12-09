# Master Makefile for dshmm

# This includes files named Rules.mk from subdirectories.  Those files
# use the variables ROOT and BUILD defined here to define source and
# destination paths.  That means that you must modify affected Rules.mk
# files if you change the directory structure.

# If subdirectories contain local Makefiles, they are for testing
# local code.  Those Makefiles are not called or used elsewhere, and
# they may be incomplete.

N_TRAIN = 50

# Look at: https://makefiletutorial.com/

ROOT:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TEX = $(ROOT)/src/TeX
PLOTSCRIPTS = $(ROOT)/src/plotscripts
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build
XFIGS = $(PLOTSCRIPTS)/xfigs
ApneaPlotScripts = $(PLOTSCRIPTS)/apnea

# Default target
$(BUILD)/TeX/skeleton/figures.pdf:

#$(BUILD)/TeX/laser/laser_fit.pdf:

## skeleton                       : Explanation of how I make each figure for the book
.PHONY : skeleton
skeleton: $(BUILD)/TeX/skeleton/figures.pdf

# Rules for making plots
include $(PLOTSCRIPTS)/basic_algorithms/Rules.mk
include $(PLOTSCRIPTS)/bounds/Rules.mk
include $(PLOTSCRIPTS)/filter/Rules.mk
include $(PLOTSCRIPTS)/laser/Rules.mk
include $(PLOTSCRIPTS)/introduction/Rules.mk
include $(PLOTSCRIPTS)/variants/Rules.mk
include $(XFIGS)/Rules.mk
include $(ApneaPlotScripts)/Rules.mk

# Rules for making data files
include $(HMMDS)/synthetic/Rules.mk
include $(HMMDS)/synthetic/filter/Rules.mk
include $(HMMDS)/synthetic/bounds/Rules.mk
include $(HMMDS)/applications/laser/Rules.mk
include $(HMMDS)/applications/apnea/Rules.mk

# Rules for making documents
include $(TEX)/skeleton/Rules.mk
include $(TEX)/laser/Rules.mk

######################Target Documents##########################################
## ds21.pdf                       : Slides for 2021 SIAM Dynamical Systems meeting
.PHONY : ds21.pdf
ds21.pdf : $(TEX)/ds21/slides.pdf

$(TEX)/ds21/slides.pdf:
	cd $(TEX)/ds21 && $(MAKE) slides.pdf

$(TEX)/bundles.pdf: $(TEX)/bundles.tex  $(INTRODUCTION_FIGS) $(BASIC_ALGORITHMS_FIGS) $(APNEA_FIGS)
	cd TeX && $(MAKE) bundles.pdf

#ToDo: Ensure that derived_data/apnea/pass1_report.pickle is up to
#date using hmmds/applications/apnea/Rules.mk
figs/pass1.pdf: plotscripts/apnea/pass1.py derived_data/apnea/pass1_report.pickle
	python $^ $@

derived_data/apnea/pass1_report.pickle:
	cd hmmds/applications/apnea && $(MAKE) pass1_report

#####################Targets for Coding Standards###############################

# test, coverage, docs_api, docs_manual

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" src/hmmds/ src/plotscripts/ tests

## check-types                    : Checks type hints
.PHONY : check-types
check-types:
	export MYPYPATH=$$PYTHONPATH; mypy --no-strict-optional hmmds/synthetic plotscripts|grep -v matplotlib|grep -v mpl_toolkits|grep -v scipy
#	export MYPYPATH=/mnt/precious/home/andy_nix/projects/proj_hmm/src; mypy --no-strict-optional hmmds/synthetic plotscripts
#	export MYPYPATH=$$PYTHONPATH; mypy --no-strict-optional hmmds/synthetic plotscripts
# --no-strict-optional allows None as default value

## lint                           : Run pylint and mypy
.PHONY : lint
lint :
	pylint --rcfile pylintrc src/hmmds/
	mypy --no-strict-optional src/hmmds/

## test                           : Run pytest on tests/
.PHONY : test
test :
	pytest tests/

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo INTRODUCTION_FIGS: $(INTRODUCTION_FIGS)
	@echo BASIC_ALGORITHMS_FIGS: $(BASIC_ALGORITHMS_FIGS)
	@echo APNEA_FIGS: $(APNEA_FIGS)
	@echo In root Makefile, ROOT: $(ROOT)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

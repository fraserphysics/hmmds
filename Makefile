N_TRAIN = 50

# Look at: https://makefiletutorial.com/

ROOT = .
XFIGS = $(ROOT)/plotscripts/xfigs
ApneaPlotScripts = $(ROOT)/plotscripts/apnea

# Default target
## software.pdf                   : Explanation of how I make each figure for the book
.PHONY : software.pdf
software.pdf: TeX/software/software.pdf

## ds21.pdf                       : Slides for 2021 SIAM Dynamical Systems meeting
.PHONY : ds21.pdf
ds21.pdf : TeX/ds21/slides.pdf

include $(XFIGS)/Rules.mk
include $(ApneaPlotScripts)/Rules.mk

TeX/software/software.pdf:
	cd TeX/software && $(MAKE) software.pdf

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

figs/Statesintro.pdf: plotscripts/stateplot.py derived_data/synthetic/states
	python  $< --data_dir derived_data/synthetic --base_name state --fig_path $@

derived_data/synthetic/states: hmmds/synthetic/StatePic.py derived_data/synthetic/m12s.4y
	python $<  derived_data/synthetic lorenz.4 lorenz.xyz m12s.4y

derived_data/synthetic/m12s.4y : hmmds/synthetic/MakeModel.py derived_data/synthetic/lorenz.xyz
	python hmmds/synthetic/MakeModel.py ${N_TRAIN} derived_data/synthetic lorenz.4 m12s.4y

derived_data/synthetic/lorenz.xyz: hmmds/synthetic/lorenz.py
	python $< --n_samples 20000 --levels 4 --quantfile derived_data/synthetic/lorenz.4 --xyzfile $@


.PHONY : test
test:
	ls $(APNEA_FIGS)

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" hmmds

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

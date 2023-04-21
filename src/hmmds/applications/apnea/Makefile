######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

ApneaDerivedData = $(ROOT)/build/derived_data/apnea
MODELS = ${ROOT}/build/derived_data/apnea/models

include Rules.mk

## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py

# I need the initial files for debugging
.PRECIOUS: $(MODELS)/a%_initial $(MODELS)/%_trained

$(MODELS)/a%_initial: model_init.py
	python $< --records a$* -- apnea_dict $@

$(MODELS)/%_trained: apnea_train.py $(MODELS)/%_initial
	python apnea_train.py --iterations 10 --records $* -- $(MODELS)/$*_initial $@

$(MODELS)/trained_all_a: $(addprefix $(MODELS)/, $(addsuffix _trained, $(ANAMES)))
	touch $@

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo MODELS: $(MODELS)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

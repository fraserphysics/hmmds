######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

ApneaDerivedData = $(ROOT)/build/derived_data/apnea
MODELS = ${ROOT}/build/derived_data/apnea/models

include Rules.mk

HR_FIGS = statistics.pdf

heart_rate.pdf: heart_rate.tex like_table.tex $(HR_FIGS)
	pdflatex $< ; pdflatex $<

like_table.tex: like_table.py $(MODELS)/unmasked_all_ac
	python like_table.py --trim_start 25 $@

statistics.pdf: statistic_plots.py
	python  statistic_plots.py $@

## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py

# Save the initial files for debugging
.PRECIOUS: $(MODELS)/a%_initial $(MODELS)/a%_masked

$(MODELS)/a%_initial: model_init.py
	python $< --records a$* --alpha_beta 1.0e3 1.0e3  --trim_start 25 apnea_dict $@

$(MODELS)/a%_masked: apnea_train.py $(MODELS)/a%_initial
	python $< --iterations 40 --records a$* --type masked --trim_start 25 $(MODELS)/a$*_initial $@

$(MODELS)/a%_declass: declass.py $(MODELS)/a%_masked
	python $^ $@

$(MODELS)/a%_unmasked: apnea_train.py $(MODELS)/a%_declass
	python $< --iterations 40 --records a$* --type unmasked --trim_start 25 $(MODELS)/a$*_declass $@


$(MODELS)/c%_initial: model_init.py
	python $< --records c$* --alpha_beta 1.0e3 1.0e3 --trim_start 25 c_model $@

$(MODELS)/c%_unmasked: apnea_train.py $(MODELS)/c%_initial
	python apnea_train.py --iterations 40 --records c$* --type unmasked $(MODELS)/c$*_initial --trim_start 25  $@

$(MODELS)/unmasked_all_ac: $(addprefix $(MODELS)/, $(addsuffix _unmasked, $(ANAMES) $(CNAMES)))
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

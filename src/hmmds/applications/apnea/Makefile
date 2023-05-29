######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

include Rules.mk

HR_FIGS = statistics.pdf

heart_rate.pdf: heart_rate.tex like_table.tex $(HR_FIGS)
	pdflatex $< ; pdflatex $<

report.txt: cross_score.py $(MODELS)/masked_all_a
	python $< --specific 1.0 > $@
#enscript -B -r -f Courier12 report.txt
report.ps: report.txt
	enscript -B -r -f Courier12 --media letter -p $@ report.txt
like_table.tex: like_table.py $(MODELS)/unmasked_all_ac
	python like_table.py --trim_start 25 $@

statistics.pdf: statistic_plots.py
	python  statistic_plots.py $@

## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py

HardSlow = a08 a10 a11 a16 a18 a20
SmallSlow = a06 a09
# Cycles in 10 minutes (2.3):
#          14  15  12  18  10  13  14  16  12  12  15  16
EasySlow = a01 a02 a03 a04 a05 a07 a12 a13 a14 a15 a17 a19
# Save the initial files for debugging
.PRECIOUS: $(MODELS)/a%_initial $(MODELS)/a%_masked

$(MODELS)/a_template_initial: model_init.py
	python $< --records $(EasySlow) --alpha_beta 1.0e3 1.0e3  --trim_start 25 fast $@

$(MODELS)/a_template_masked: apnea_train.py $(MODELS)/a_template_initial
	python $< --iterations 40 --records $(EasySlow) --type masked --trim_start 25 $(MODELS)/a_template_initial $@

$(MODELS)/a%_initial: model_init.py
	python $< --records a$* --alpha_beta 1.0e3 1.0e3  --trim_start 25 apnea_dict $@

$(MODELS)/a2_etc_masked: apnea_train.py $(MODELS)/a02_initial
	python $< --iterations 40 --records a02 a05 a06 a09 a11 a16 a18 a19 a20 --type masked --trim_start 25 $(MODELS)/a02_initial $@

$(MODELS)/a3_etc_masked: apnea_train.py $(MODELS)/a03_initial
	python $< --iterations 40 --records a03 a07 a09 a17 a18 --type masked --trim_start 25 $(MODELS)/a03_initial $@

$(MODELS)/a1_etc_masked: apnea_train.py $(MODELS)/a01_initial
	python $< --iterations 40 --records a01 a04 a08 a13 a16 --type masked --trim_start 25 $(MODELS)/a01_initial $@

$(MODELS)/a%_masked: apnea_train.py $(MODELS)/a%_initial
	python $< --iterations 40 --records a$* --type masked --trim_start 25 $(MODELS)/a$*_initial $@

$(MODELS)/%_small: model_init.py
	python $< --records $* --alpha_beta 1.0e3 1.0e3 --trim_start 25 c_model $@

$(MODELS)/%_unmasked: apnea_train.py $(MODELS)/%_small
	python apnea_train.py --iterations 40 --records $* --type unmasked $(MODELS)/$*_small --trim_start 25  $@

$(MODELS)/unmasked_all_ac: $(addprefix $(MODELS)/, $(addsuffix _unmasked, $(ANAMES) $(CNAMES)))
	touch $@

$(MODELS)/masked_all_a: $(addprefix $(MODELS)/, $(addsuffix _masked, $(ANAMES)))
	touch $@

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

.PHONY : clean
clean:
	rm -f *.aux *.log *.pdf like_table.tex

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

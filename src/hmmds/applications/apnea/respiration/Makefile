######################Apnea#########################################
# Several paths and parameters are shared in instances of utilities.args

ROOT = ../../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

include ../Rules.mk

APLUSNAMES = $(ANAMES) b01 b02 b03 b04 c08 c10

errors_vs_ar_order.pdf: compare_models.py $(addprefix $(MODELS)/, $(AR_MODELS))
	python compare_models.py --models $(AR_MODELS) --parameters $(AR_ORDERS) \
--parameter_name "AR order" --records $(APLUSNAMES) -- $@ > errors_vs_ar_order.txt

score.txt: score.py pass2.out
	python score.py pass2.out $@

score.tex: score.py pass2.out
	python score.py --tex pass2.out $@

pass1.out: pass1.py
	python $< --trim 20 --threshold .36 --model $(BEST_MODEL) $@

pass2.out: pass2.py pass1.out $(BEST_MODEL) $(MODELS)/c_model
	python $< --model_paths $(MODELS)/c_model $(BEST_MODEL) pass1.out temp
	cp temp $@

# norm_power0.9threshold-12ar%prom6.1_masked
NORM_AR_MODELS = $(addsuffix prom$(NORM_PROM)_masked, \
$(addprefix norm_power$(NORM_POWER)threshold$(NORM_LOG_THRESHOLD)ar, $(AR_ORDERS)))

# Save the initial files for debugging
.PRECIOUS: $(MODELS)/%_initial $(MODELS)/%_masked

# config6.pkl is not used for c_model, but model_init.py requires a
# config argument
$(MODELS)/c_model: model_init.py config6.pkl
	python $< config6.pkl c_model $@

#$(MODELS)/respiration2_initial: model_init.py
lphr_respiration2_initial: model_init.py
	python $< --heart_rate_sample_frequency 6 --low_pass_period 2 --band_pass_center  16 --band_pass_width 4 --AR_order 6 lphr_respiration2 $@

#$(MODELS)/%_masked: apnea_train.py $(MODELS)/%_initial
%_masked: ../apnea_train.py %_initial
	python $< --records $(APLUSNAMES) --iterations 2 $*_initial $@

## yapf                          : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

.PHONY : clean
clean:
	rm -f *.aux *.log *.pdf like_table.tex *.pkl $(MODELS)/* *flag

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo APLUSNAMES: $(APLUSNAMES)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

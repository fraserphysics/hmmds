######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

# default target
heart_rate.pdf:

include Rules.mk

# Defaults copied from utilities.py
DEFAULT_ARGS = --root $(ROOT) \
--derived_apnea_data build/derived_data/apnea \
--model_dir build/derived_data/apnea/models \
--heart_rate_path_format build/derived_data/ECG/{0}_self_AR3/heart_rate \
--rtimes raw_data/Rtimes \
--expert raw_data/apnea/summary_of_training \
--iterations 10 \
--heart_rate_sample_frequency 24  \
--power_and_threshold 1.6, 1.0  \
--trim_start 0 \
--fft_width 4096 \
--low_pass_period 15.0 \
--band_pass_center 14.0


# Optimize pass2 parameters over files classified as A or near
# boundary by pass1.
APLUSNAMES = $(ANAMES) b01 b02 b03 b04 c08 c10

BEST_MODEL = $(MODELS)/two_ar5_masked
MIN_PROMINENCE = 6.0

# Change characteristics to config
config.pkl: config_stats.py
	python $< --records $(APLUSNAMES) -- $(MIN_PROMINENCE) $@

config%.pkl: config_stats.py
	python $< --records $(APLUSNAMES) -- $* $@

interval_pdfs.pdf: apnea_ratio.py config.pkl
	python $< --records $(APLUSNAMES) -- config.pkl $@

analyze_peaks.pdf: analyze_peaks.py config.pkl
	python $^ --figure_path $@

NORM_MODEL = $(MODELS)/norm_ar5_masked

AR_ORDERS = 1 2 3 4 5 6 7 8 9 10
AR_MODELS = $(addsuffix _masked, $(addprefix two_ar, $(AR_ORDERS)))
NORM_AR_MODELS = $(addsuffix _masked, $(addprefix norm_ar, $(AR_ORDERS)))
errors_vs_ar_order.pdf: compare_models.py $(addprefix $(MODELS)/, $(AR_MODELS))
	python compare_models.py --models $(AR_MODELS) --parameters $(AR_ORDERS) \
--parameter_name "AR order" --records $(APLUSNAMES) -- $@ > errors_vs_ar_order.txt

power_threshold.pdf: survey_power_threshold.py $(BEST_MODEL)
	python $< --records $(APLUSNAMES) -- $(BEST_MODEL) $@ > power_threshold.txt

PROMINENCES = 3 3.5 4 4.5 5 5.5 5.7 5.8 5.9 5.95 6 6.05 6.1 6.2 6.3 6.4 6.5 7 7.5 8
PROMINENCE_MODELS = $(addprefix $(BEST_MODEL), $(PROMINENCES))
prominence_models: $(PROMINENCE_MODELS)
	touch $@
prominence_study.pdf: prominence_study.py prominence_models $(PROMINENCE_MODELS)
	python $< --prominences $(PROMINENCES) --template $(BEST_MODEL)% --records $(APLUSNAMES) --fig_path $@ > prominence_study.txt

HR_FIGS = statistics.pdf errors_vs_ar_order.pdf power_threshold.pdf prominence_study.pdf \
interval_pdfs.pdf analyze_peaks.pdf norm_power_threshold.pdf norm_errors_vs_ar_order.pdf

heart_rate.pdf: heart_rate.tex  $(HR_FIGS) two.tex score.tex lengths.txt norm_score.tex
	pdflatex $< ; pdflatex $<

pass1.out: pass1.py
	python $< --trim 20 --threshold .36 --model $(BEST_MODEL) $@

pass2.out: pass2.py pass1.out $(BEST_MODEL) $(MODELS)/c_model
	python $< --model_paths ../../../../build/derived_data/apnea/models/c_model $(BEST_MODEL) pass1.out temp
	cp temp $@

score.txt: score.py pass2.out
	python score.py pass2.out $@

score.tex: score.py pass2.out
	python score.py --tex pass2.out $@

norm_errors_vs_ar_order.pdf: compare_models.py $(addprefix $(MODELS)/, $(NORM_AR_MODELS))
	python compare_models.py --models $(NORM_AR_MODELS) --parameters $(AR_ORDERS) \
--parameter_name "AR order" --records $(APLUSNAMES) --power_and_threshold 1 1 $@ > norm_errors_vs_ar_order.txt

norm_pass2.out: pass2.py pass1.out $(NORM_MODEL) $(MODELS)/c_model
	python $< --model_paths ../../../../build/derived_data/apnea/models/c_model $(NORM_MODEL) --power_and_threshold 1 1 pass1.out temp
	cp temp $@

norm_score.tex: score.py norm_pass2.out
	python score.py --tex --power_and_threshold 1 1 norm_pass2.out $@

norm_power_threshold.pdf: survey_power_threshold.py $(NORM_MODEL)
	python $< --records $(APLUSNAMES) --power_and_threshold 1 1  $(NORM_MODEL) $@ > norm_power_threshold.txt

# Lengths of time series for each training record
lengths.txt: test_alignment.py $(BEST_MODEL)
	python $< --model $(BEST_MODEL) > $@

# A table of pass2 performance on each a-file.
two.tex: prominence_study.py $(BEST_MODEL) $(BEST_MODEL)$(MIN_PROMINENCE)
	python $< --prominences $(MIN_PROMINENCE) --template $(MODELS)/two_ar5_masked% --latex $@

report.txt: cross_score.py $(MODELS)/masked_all_a
	python $< --specific 1.0 > $@
#enscript -B -r -f Courier12 report.txt
report.ps: report.txt
	enscript -B -r -f Courier12 --media letter -p $@ report.txt
like_table.tex: like_table.py $(MODELS)/unmasked_all_ac
	python like_table.py --trim_start 25 $@

NORM_MODELS = $(addsuffix _masked, $(addprefix norm_ar, $(AR_ORDERS)))

statistics.pdf: statistic_plots.py
	python statistic_plots.py --trim 20 --AR_order 6 $@

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

# config.pkl is not used
$(MODELS)/c_model: model_init.py config.pkl
	python $< config.pkl c_model $@

$(MODELS)/two_ar5_initial%: model_init.py config%.pkl
	python $< --low_pass_period 8 \
--records $(ANAMES) --alpha_beta 50.0 10.0 --AR_order 5 \
config$*.pkl two_intervals $@

$(MODELS)/two_ar5_masked%: apnea_train.py $(MODELS)/two_ar5_initial%
	python $< --records $(ANAMES) -- $(MODELS)/two_ar5_initial$* $@

$(MODELS)/two_ar%_initial: model_init.py config.pkl
	python $<  --low_pass_period 8 \
--records $(APLUSNAMES) --alpha_beta 50.0 10.0 --AR_order $* \
 config.pkl two_intervals $@

$(MODELS)/two_ar%_masked: apnea_train.py $(MODELS)/two_ar%_initial
	python $<  --records $(APLUSNAMES) -- $(MODELS)/two_ar$*_initial $@

$(MODELS)/norm_ar%_initial: model_init.py config.pkl
	python $<  --low_pass_period 8 \
--records $(APLUSNAMES) --alpha_beta 50.0 10.0 --AR_order $* \
--power_and_threshold 1 1 config.pkl two_normalized $@

$(MODELS)/norm_ar%_masked: apnea_train.py $(MODELS)/norm_ar%_initial
	python $<  --records $(APLUSNAMES) --iterations 10 --power_and_threshold 1 1 $(MODELS)/norm_ar$*_initial $@

$(MODELS)/intervals_ar%_initial: model_init.py config.pkl
	python $<  --low_pass_period 8 \
--records $(EasySlow) --alpha_beta 50.0 10.0 --AR_order $* \
config.pkl hmm_intervals $@

$(MODELS)/intervals_ar%_masked: apnea_train.py $(MODELS)/intervals_ar%_initial
	python $<  --records $(ANAMES) $(MODELS)/intervals_ar$*_initial $@

$(MODELS)/%_small: model_init.py
	python $< --records $* --alpha_beta 1.0e3 1.0e3 --trim_start 25 c_model $@

$(MODELS)/%_unmasked: apnea_train.py $(MODELS)/%_small
	python apnea_train.py --iterations 40 --records $* $(MODELS)/$*_small --trim_start 25 $@

$(MODELS)/unmasked_all_ac: $(addprefix $(MODELS)/, $(addsuffix _unmasked, $(ANAMES) $(CNAMES)))
	touch $@

$(MODELS)/masked_all_a: $(addprefix $(MODELS)/, $(addsuffix _masked, $(ANAMES)))
	touch $@

## yapf                          : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

.PHONY : clean
clean:
	rm -f *.aux *.log *.pdf like_table.tex

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo AR_MODELS: $(AR_MODELS)
	@echo MODELS: $(MODELS)
	@echo PROMINENCES: $(PROMINENCES)
	@echo PROMINENCE_MODELS: $(PROMINENCE_MODELS)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

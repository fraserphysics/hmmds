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
--low_pass_period 8.0 \
--band_pass_center 14.0

# In model_init.py alpha_beta = 50 10

# Optimize pass2 parameters over these files classified as A or near
# boundary by pass1.
APLUSNAMES = $(ANAMES) b01 b02 b03 b04 c08 c10

###########################Two Chain Unnormalized###############################
POWER = 1.0
LOG_THRESHOLD = -10
POWER_THRESHOLD = --power_and_threshold $(POWER) 1.0e$(LOG_THRESHOLD)
AR_ORDER = 5
PROM = 4
BEST_MODEL = $(MODELS)/two_power$(POWER)threshold$(LOG_THRESHOLD)ar$(AR_ORDER)prom$(PROM)_masked

AR_ORDERS = 1 2 3 4 5 6 7 8 9 10
AR_MODELS = $(addsuffix prom$(PROM)_masked, \
$(addprefix two_power$(POWER)threshold$(LOG_THRESHOLD)ar, $(AR_ORDERS)))
PROMINENCES = 3 3.5 4 4.5 5 5.5 5.7 5.8 5.9 5.95 6 6.05 6.1 6.2 6.3 6.4 6.5 7 7.5 8
PROMINENCE_MODELS = $(addsuffix _masked, $(addprefix \
    $(MODELS)/two_power$(POWER)threshold$(LOG_THRESHOLD)ar$(AR_ORDER)prom, \
    $(PROMINENCES)))

errors_vs_ar_order.pdf: compare_models.py $(addprefix $(MODELS)/, $(AR_MODELS))
	python compare_models.py --models $(AR_MODELS) --parameters $(AR_ORDERS) \
--parameter_name "AR order" --records $(APLUSNAMES) -- $@ > errors_vs_ar_order.txt

power_threshold.pdf: survey_power_threshold.py $(BEST_MODEL)
	python $< --powers 0.0 3.5 36 --thresholds -20.0 20.0 21 --records $(APLUSNAMES) -- $(BEST_MODEL) $@ > power_threshold.txt

prominence_study.pdf: prominence_study.py $(PROMINENCE_MODELS)
	python $< --prominences $(PROMINENCES) --template \
    $(MODELS)/two_power$(POWER)threshold$(LOG_THRESHOLD)ar$(AR_ORDER)prom%_masked \
    --records $(APLUSNAMES) --fig_path $@ > prominence_study.txt

score.txt: score.py pass2.out
	python score.py pass2.out $@

score.tex: score.py pass2.out
	python score.py --tex pass2.out $@

pass1.out: pass1.py
	python $< --trim 20 --threshold .36 --model $(BEST_MODEL) $@

pass2.out: pass2.py pass1.out $(BEST_MODEL) $(MODELS)/c_model
	python $< --model_paths $(MODELS)/c_model $(BEST_MODEL) pass1.out temp
	cp temp $@

###########################Two Chain Normalized#################################
NORM_POWER = 1.5
NORM_LOG_THRESHOLD = -6
NORM_POWER_THRESHOLD = --power_and_threshold $(NORM_POWER) 1.0e$(NORM_LOG_THRESHOLD)
NORM_AR = 6
NORM_PROM = 3.3
NORM_MODEL = $(MODELS)/norm_power$(NORM_POWER)threshold$(NORM_LOG_THRESHOLD)ar$(NORM_AR)prom$(NORM_PROM)_masked

NORM_PROMINENCES = 1.5 2 2.5 3 3.1 3.2 3.3 3.4 3.5 4 4.5 5

# Power Threshold AR Prominence Norm Errors
# 1.5   1.0e-2    3  5.0        2623
# 1.1   1.0e-6    4  5.0        2643
# 0.9   1.0e-12   5  6.1        2539
########### Start using make_init.py ######
# 1.5   1.0e-2    3  3.3        2825
# 0.9   1.0e-12   6  6.1        2834
# 0.9   1.0e-12   5  6.1        2869
# 0.9   1.0e-12   5  3.2        2743
# 1.6   1.0e-10   7  3.2        2679
# 1.5   1.0e-6    6  3.3        2660

# norm_power0.9threshold-12ar%prom6.1_masked
NORM_AR_MODELS = $(addsuffix prom$(NORM_PROM)_masked, \
$(addprefix norm_power$(NORM_POWER)threshold$(NORM_LOG_THRESHOLD)ar, $(AR_ORDERS)))

# norm_power0.9threshold-12ar6prom%_masked
NORM_PROMINENCE_MODELS = $(addsuffix _masked, $(addprefix \
$(MODELS)/norm_power$(NORM_POWER)threshold$(NORM_LOG_THRESHOLD)ar$(NORM_AR)prom, \
$(NORM_PROMINENCES)))

norm_analyze_peaks.pdf: analyze_peaks.py norm_config$(NORM_PROM).pkl
	python $^ --normalize --records $(APLUSNAMES) --figure_path $@

norm_interval_pdfs.pdf: apnea_ratio.py norm_config$(NORM_PROM).pkl
	python $< --records $(APLUSNAMES) --normalize norm_config$(NORM_PROM).pkl $@

norm_prominence_study.pdf: prominence_study.py $(NORM_PROMINENCE_MODELS)
	python $< --prominences $(NORM_PROMINENCES) --template \
    $(MODELS)/norm_power$(NORM_POWER)threshold$(NORM_LOG_THRESHOLD)ar$(NORM_AR)prom%_masked \
    --records $(APLUSNAMES) --fig_path $@ > norm_prominence_study.txt

norm_errors_vs_ar_order.pdf: compare_models.py $(addprefix $(MODELS)/, $(NORM_AR_MODELS))
	python compare_models.py --models $(NORM_AR_MODELS) --parameters $(AR_ORDERS) \
    --parameter_name "AR order" --records $(APLUSNAMES) \
    $(NORM_POWER_THRESHOLD) $@ > norm_errors_vs_ar_order.txt

norm_power_threshold.pdf: survey_power_threshold.py $(NORM_MODEL)
	python $< --powers 0.0 3.5 36 --thresholds -20.0 20.0 21 --records $(APLUSNAMES) \
    $(NORM_POWER_THRESHOLD) $(NORM_MODEL) $@ > norm_power_threshold.txt

norm_score.tex: score.py norm_pass2.out
	python score.py --tex $(NORM_POWER_THRESHOLD) norm_pass2.out $@

norm_pass2.out: pass2.py pass1.out $(NORM_MODEL) $(MODELS)/c_model
	python $< --model_paths ../../../../build/derived_data/apnea/models/c_model \
    $(NORM_MODEL) $(NORM_POWER_THRESHOLD) pass1.out temp
	cp temp $@

lphr_respiration_pass2.out: pass2.py pass1.out $(MODELS)/c_model $(MODELS)/lphr_respiration2_masked
	python $< --model_paths $(MODELS)/c_model $(MODELS)/lphr_respiration2_masked \
    --power_and_threshold 1.5 0.5 pass1.out temp
	cp temp $@

lphr_respiration_score.tex: score.py lphr_respiration_pass2.out
	python $^ $@  --tex

#####################Calculate statistics of peaks##############################
# config.pkl holds statistics derived from detected peaks
config.pkl: config_stats.py
	python $< --records $(APLUSNAMES) -- $(PROM) $@

# % is minimum threshold for detecting a prominence
config%.pkl: config_stats.py
	python $< --records $(APLUSNAMES) -- $* $@

norm_config%.pkl: config_stats.py config%.pkl
	python $< --records $(APLUSNAMES) --normalize $* $@

interval_pdfs.pdf: apnea_ratio.py config6.pkl
	python $< --records $(APLUSNAMES) -- config6.pkl $@

analyze_peaks.pdf: analyze_peaks.py config6.pkl
	python $^ --records $(APLUSNAMES) --figure_path $@

########################Build heart_rate.pdf####################################
# HR_INCLUDES are items that are not in the INCLUDE_LIST pattern
HR_INCLUDES = statistics.pdf errors_vs_ar_order.pdf power_threshold.pdf \
prominence_study.pdf interval_pdfs.pdf norm_interval_pdfs.pdf analyze_peaks.pdf \
norm_analyze_peaks.pdf norm_power_threshold.pdf score.tex lengths.txt \
lphr_respiration_score.tex

# INCUDE_LIST matches StudyFigs in heart_rate.tex
INCLUDE_LIST = _prominence_study.pdf _power_threshold.pdf _errors_vs_ar_order.pdf _score.tex
NORM_INCLUDES = $(addprefix norm, $(INCLUDE_LIST))

heart_rate.pdf: heart_rate.tex  $(HR_INCLUDES) $(NORM_INCLUDES)
	pdflatex $< ; pdflatex $<

################################################################################

# Lengths of time series for each training record
lengths.txt: test_alignment.py $(BEST_MODEL)
	python $< --model $(BEST_MODEL) > $@

report.txt: cross_score.py $(MODELS)/masked_all_a
	python $< --specific 1.0 > $@
#enscript -B -r -f Courier12 report.txt
report.ps: report.txt
	enscript -B -r -f Courier12 --media letter -p $@ report.txt
like_table.tex: like_table.py $(MODELS)/unmasked_all_ac
	python like_table.py --trim_start 25 $@

statistics.pdf: statistic_plots.py $(BEST_MODEL)
	python statistic_plots.py --trim 20 --model $(BEST_MODEL) $@

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
.PRECIOUS: $(MODELS)/%_initial $(MODELS)/%_masked

# new_config.pkl is not used for c_model, but model_init.py requires a
# config argument
$(MODELS)/c_model: model_init.py new_config.pkl
	python $< new_config.pkl c_model $@

$(MODELS)/two_%_initial: model_init.py config_stats.py make_init.py
	python make_init.py two_intervals $* $@

# EG If % were power1.5threshold-12ar5prom6.1, the code would issue
# the following: python model_init.py --power_and_threshold 1.5
# 1.0e-12 --AR_order 5 norm_config6.1.pkl two_normalized
# ../../../../build/derived_data/apnea/models/norm_power1.5threshold-12ar5prom6.1_initial
$(MODELS)/norm_%_initial: model_init.py config_stats.py make_init.py
	python make_init.py two_normalized $* $@

$(MODELS)/lphr_respiration2_initial: model_init.py config4.pkl
	python $< --heart_rate_sample_frequency 6 --low_pass_period 2 --band_pass_center  16 --band_pass_width 4 --AR_order 6 config4.pkl lphr_respiration2 $@

new_config.pkl: config_stats.py
	python $< --normalize 4.0 $@

$(MODELS)/two_varg_initial: model_init.py new_config.pkl
	mkdir -p $(MODELS)
	python $< --model_sample_frequency 8 --low_pass_period 2 --band_pass_center  16 --band_pass_width 4 --AR_order 6 new_config.pkl two_varg $@

$(MODELS)/%_masked: apnea_train.py $(MODELS)/%_initial
	python $< --records $(APLUSNAMES) --iterations 10 $(MODELS)/$*_initial $@

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
	@echo MODELS: $(MODELS)
	@echo BEST_MODEL: $(BEST_MODEL)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

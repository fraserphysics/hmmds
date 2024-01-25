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
--heart_rate_sample_frequency 120  \
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
POWER_DICT = --power_dict slow 1.0 peak 1.0 interval $(POWER) class 1.0
LOG_THRESHOLD = -10
POWER_THRESHOLD = $(POWER_DICT) --threshold 1.0e$(LOG_THRESHOLD)
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
	python $< --powers 0.0 3.5 36 --thresholds -20.0 20.0 21 $(POWER_DICT) \
 --records $(APLUSNAMES) -- $(BEST_MODEL) $@ > power_threshold.txt

prominence_study.pdf: prominence_study.py $(PROMINENCE_MODELS)
	python $< $(POWER_THRESHOLD) --prominences $(PROMINENCES) --template \
    $(MODELS)/two_power$(POWER)threshold$(LOG_THRESHOLD)ar$(AR_ORDER)prom%_masked \
    --records $(APLUSNAMES) --fig_path $@ > prominence_study.txt

score.txt: score.py pass2.out
	python score.py pass2.out $@

score.tex: score.py pass2.out
	python score.py --tex pass2.out $@

pass1.out: pass1.py
	python $< --trim 20 --border .36 --model $(BEST_MODEL) $@

pass2.out: pass2.py pass1.out $(BEST_MODEL) $(MODELS)/c_model
	python $< --model_paths $(MODELS)/c_model $(BEST_MODEL) pass1.out temp
	cp temp $@

###########################Two Chain Normalized#################################
NORM_POWER = 1.5
NORM_POWER_DICT = --power_dict slow 1.0 peak 1.0 interval $(NORM_POWER) class 1
NORM_LOG_THRESHOLD = -6
NORM_POWER_THRESHOLD = $(NORM_POWER_DICT) --threshold 1.0e$(NORM_LOG_THRESHOLD)
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
	python $< --records $(APLUSNAMES) -- norm_config$(NORM_PROM).pkl $@

norm_prominence_study.pdf: prominence_study.py $(NORM_PROMINENCE_MODELS)
	python $< $(NORM_POWER_THRESHOLD) --template \
    $(MODELS)/norm_power$(NORM_POWER)threshold$(NORM_LOG_THRESHOLD)ar$(NORM_AR)prom%_masked \
    --records $(APLUSNAMES) --prominences $(NORM_PROMINENCES) --fig_path $@ \
    > norm_prominence_study.txt

norm_errors_vs_ar_order.pdf: compare_models.py $(addprefix $(MODELS)/, $(NORM_AR_MODELS))
	python compare_models.py --models $(NORM_AR_MODELS) --parameters $(AR_ORDERS) \
    --parameter_name "AR order" --records $(APLUSNAMES) \
    $(NORM_POWER_THRESHOLD) $@ > norm_errors_vs_ar_order.txt

norm_power_threshold.pdf: survey_power_threshold.py $(NORM_MODEL)
	python $< --powers 0.0 3.5 36 --components slow peak interval class \
  --thresholds -20.0 20.0 21 --records $(APLUSNAMES) $(NORM_POWER_THRESHOLD) \
 $(NORM_MODEL) $@ > norm_power_threshold.txt

norm_score.tex: score.py norm_pass2.out
	python score.py --tex $(NORM_POWER_THRESHOLD) norm_pass2.out $@

norm_pass2.out: pass2.py pass1.out $(NORM_MODEL) $(MODELS)/c_model
	python $< --model_paths ../../../../build/derived_data/apnea/models/c_model \
    $(NORM_MODEL) $(NORM_POWER_THRESHOLD)  --components slow peak interval class -- pass1.out temp
	cp temp $@

# FixMe: Rename varg2chain to v2c
# x_varg2chain has fixed parameters.  The x_ distinguishes it from
# varg2chain% in which % describes a collection of parameters.
x_varg2chain_pass2.out: pass2.py pass1.out $(MODELS)/c_model $(MODELS)/x_varg2chain_masked
	python $< --model_paths $(MODELS)/c_model $(MODELS)/x_varg2chain_masked \
    --power_dict slow 1.0 peak 0.0 interval 0.0 class 1.0 --threshold 0.5 pass1.out temp
	cp temp $@

x_varg2chain_score.tex: score.py x_varg2chain_pass2.out
	python $^ $@  --tex

#####################Calculate statistics of peaks##############################
# config.pkl holds statistics derived from detected peaks
config.pkl: config_stats.py
	python $< --records $(APLUSNAMES) -- $(PROM) $@

# % is minimum threshold for detecting a prominence
config%.pkl: config_stats.py
	python $< --records $(APLUSNAMES) -- $* $@

norm_config%.pkl: config_stats.py
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
varg2state_score.tex

# INCUDE_LIST matches StudyFigs in heart_rate.tex
INCLUDE_LIST = _prominence_study.pdf _power_threshold.pdf _errors_vs_ar_order.pdf _score.tex
NORM_INCLUDES = $(addprefix norm, $(INCLUDE_LIST))

heart_rate.pdf: heart_rate.tex  $(HR_INCLUDES) $(NORM_INCLUDES)
	pdflatex $< ; pdflatex $<

########################Build hand_opt.pdf####################################

LIST_ERRORS = pt ip ar fs lpp rc rw rs
v4sERRORS = $(addsuffix .pdf, $(addprefix v4s_errors_vs_, $(LIST_ERRORS)))
hand_opt.pdf: hand_opt.tex $(v4sERRORS) v4s_threshold.pdf v4s_score.tex
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

new_config.pkl: config_stats.py
	python $< --normalize 4.0 $@

# new_config.pkl is not used for c_model, but model_init.py requires a
# config argument
$(MODELS)/c_model: model_init.py new_config.pkl
	python $< --model_sample_frequency 12 new_config.pkl c_model $@

$(MODELS)/two_%_initial: model_init.py config_stats.py make_init.py
	python make_init.py two_intervals $* $@

$(MODELS)/norm_%_initial: model_init.py config_stats.py make_init.py
	python make_init.py two_normalized $* $@

$(MODELS)/varg2state_initial: model_init.py config4.pkl
	python $< --model_sample_frequency 6 --low_pass_period 2 --band_pass_center  16 --band_pass_width 4 --AR_order 6 config4.pkl varg2state $@

varg2state_pass2.out: pass2.py pass1.out $(MODELS)/c_model $(MODELS)/varg2state_masked
	python $< --model_paths $(MODELS)/c_model $(MODELS)/varg2state_masked \
    --power_dict hr_respiration 1.5 class 1.0 --threshold 0.5 pass1.out temp
	cp temp $@

varg2state_score.tex: score.py varg2state_pass2.out
	python $^ $@  --tex

varg_power_threshold.pdf: survey_power_threshold.py $(VARG_MODEL)
	python $< --thresholds -80.0 0.0 21 --records $(APLUSNAMES) \
    $(VARG_POWER_THRESHOLD) $(VARG_MODEL) $@ > varg_power_threshold.txt


###################### 2 states with only VARG observation, V2S ###########

# AutoRegressive order
V2S_AR = 16
# Model sample frequency cpm
V2S_FS = 4
# Low Pass Period seconds
V2S_LPP = 10
# Respiration Center frequency cpm
V2S_RC = 17
# Respiration Width cpm
V2S_RW = 4.7
# Filter for Respiration Smoothing in cpm.
V2S_RS = .44

V2S_BEST = $(MODELS)/varg2state_ar$(V2S_AR)fs$(V2S_FS)lpp$(V2S_LPP)rc$(V2S_RC)rw$(V2S_RW)rs$(V2S_RS)_masked

# Default threshold in utilities.py of log(1) = 0 is good
v2s_threshold.pdf: survey_power_threshold.py $(V2S_BEST)
	python $< --thresholds -1.0 1.0 21 --records $(APLUSNAMES) -- $(V2S_BEST) $@ \
    > v2s_threshold.txt

# Sensitivity to AutoRegressive order
V2S_ARs = 5 10 15 16 17 18 19 20 25 30
VAR_MODELS = $(addsuffix fs$(V2S_FS)lpp$(V2S_LPP)rc$(V2S_RC)rw$(V2S_RW)rs$(V2S_RS)_masked, \
    $(addprefix varg2state_ar, $(V2S_ARs)))

v_errors_vs_ar.pdf: compare_models.py $(addprefix $(MODELS)/, $(VAR_MODELS))
	python compare_models.py --models $(VAR_MODELS) --parameters $(V2S_ARs) \
--parameter_name "AR order" --records $(APLUSNAMES) -- $@ > v_errors_vs_ar.txt

# Sensitivity to Sample Frequency
V2S_FSs = 2 3 4 5 6
VFS_MODELS = $(addsuffix lpp$(V2S_LPP)rc$(V2S_RC)rw$(V2S_RW)rs$(V2S_RS)_masked, \
    $(addprefix varg2state_ar$(V2S_AR)fs, $(V2S_FSs)))

v_errors_vs_fs.pdf: compare_models.py $(addprefix $(MODELS)/, $(VFS_MODELS))
	python compare_models.py --models $(VFS_MODELS) --parameters $(V2S_FSs) \
--parameter_name "Sample Frequency" --records $(APLUSNAMES) -- $@ > v_errors_vs_fs.txt

# Sensitivity to Low Pass Period in seconds.
V2S_LPPs = 5 8 9 10 11 12 15 20 25 30 40 45 50
LPP_MODELS = $(addsuffix rc$(V2S_RC)rw$(V2S_RW)rs$(V2S_RS)_masked, $(addprefix \
    varg2state_ar$(V2S_AR)fs$(V2S_FS)lpp, \
    $(V2S_LPPs)))

v_errors_vs_lpp.pdf: compare_models.py $(addprefix $(MODELS)/, $(LPP_MODELS))
	python compare_models.py --models $(LPP_MODELS) --parameters $(V2S_LPPs) \
--parameter_name "Low Pass Period" --records $(APLUSNAMES) -- $@ > v_errors_vs_lpp.txt

# Sensitivity to Respiration Center frequency in cpm
V2S_RCs = 12 14 16 17 18 19 20 22 24 28
RC_MODELS = $(addsuffix rw$(V2S_RW)rs$(V2S_RS)_masked, $(addprefix \
    varg2state_ar$(V2S_AR)fs$(V2S_FS)lpp$(V2S_LPP)rc, \
    $(V2S_RCs)))

v_errors_vs_rc.pdf: compare_models.py $(addprefix $(MODELS)/, $(RC_MODELS))
	python compare_models.py --models $(RC_MODELS) --parameters $(V2S_RCs) \
--parameter_name "Center Frequency" --records $(APLUSNAMES) -- $@ > v_errors_vs_rc.txt

# Sensitivity to Respiration width frequency in cpm
V2S_RWs = 4 4.3 4.4 4.5 4.6 4.7 5 5.1
RW_MODELS = $(addsuffix rs$(V2S_RS)_masked, $(addprefix \
    varg2state_ar$(V2S_AR)fs$(V2S_FS)lpp$(V2S_LPP)rc$(V2S_RC)rw, \
    $(V2S_RWs)))

v_errors_vs_rw.pdf: compare_models.py $(addprefix $(MODELS)/, $(RW_MODELS))
	python compare_models.py --models $(RW_MODELS) --parameters $(V2S_RWs) \
--parameter_name "Frequency Width" --records $(APLUSNAMES) -- $@ > v_errors_vs_rw.txt

# Sensitivity to Respiration Smoothing frequency in cpm
V2S_RSs =  .2 .3 .4 .41 .42 .43 .44 .45 .46 .6 1.0
RS_MODELS = $(addsuffix _masked, $(addprefix \
    varg2state_ar$(V2S_AR)fs$(V2S_FS)lpp$(V2S_LPP)rc$(V2S_RC)rw$(V2S_RW)rs, \
    $(V2S_RSs)))

v_errors_vs_rs.pdf: compare_models.py $(addprefix $(MODELS)/, $(RS_MODELS))
	python compare_models.py --models $(RS_MODELS) --parameters $(V2S_RSs) \
    --parameter_name "Respiration Smoothing Filter" --records $(APLUSNAMES) -- \
    $@ > v_errors_vs_rs.txt

# EG $(MODELS)/varg2state_ar16fs4lpp10rc17rw4.7rs.44_init

$(MODELS)/varg2state_%_init: make_init.py model_init.py config_stats.py
	python $< varg2state $* $@

$(MODELS)/varg2state%_masked: apnea_train.py $(MODELS)/varg2state%_init
	python $< --records $(APLUSNAMES) --iterations 3 $(MODELS)/varg2state$*_init $@

v2s_pass2.out: pass2.py pass1.out $(MODELS)/c_model $(V2S_BEST)
	python $< --model_paths $(MODELS)/c_model $(V2S_BEST) \
    --threshold 1.0 pass1.out temp
	cp temp $@

v2s_score.tex: score.py v2s_pass2.out
	python $^ $@  --tex

########### 4 states with VARG and interval observation, V4S ###########

# With only VARG:         N->A: 1805 A->N: 1036 N_error: 2841 P_error: 0.221

# With VARG and interval: 4.8 N->A: 1748 A->N: 1071 N_error: 2819 P_error: 0.219

# AutoRegressive order
V4S_AR = 9
# Model sample frequency cpm
V4S_FS = 6
# Low Pass Period seconds
V4S_LPP = 58.0
# Respiration Center frequency cpm
V4S_RC = 12.6
# Respiration Width cpm
V4S_RW = 4.9
# Filter for Respiration Smoothing in cpm.
V4S_RS = .458
# Prominence Threshold
V4S_PT = 5.0
# Exponential weight for varg component
V4S_VP = 1.0
# Exponential weight for interval component
V4S_IP = 0.34
# Detection threshold
V4S_THRESHOLD = 93

V4S_BEST = $(MODELS)/varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked
# Yields 3072 errors  Earlier got 2819

v4s_opt: optimize.py
	python $< --records  $(APLUSNAMES) -- $@

COMPARE = python compare_models.py --n_apnea 6446 --records $(APLUSNAMES) --threshold $(V4S_THRESHOLD)

v4s_threshold.pdf: survey_power_threshold.py $(V4S_BEST)
	python $< --thresholds 1.5 2.5 21 --powers 0.3 0.4 21 \
 --power_dict hr_respiration 1.0 interval $(V4S_IP) class 1.0 --records $(APLUSNAMES) \
 --threshold $(V4S_THRESHOLD) $(V4S_BEST) $@ > v4s_threshold.txt

# Sensitivity to PT, Prominence Threshold for detecting peaks
V4S_PTs = 3.4 3.6 3.8 4.0 4.1 4.2 4.3 4.4 4.6 4.8 4.9 5.0 5.1 5.2 5.4 5.6 5.8 6.0 6.2 6.4 6.6 6.8 7.0
V4S_PT_MODELS = $(addsuffix vp$(V4S_VP)ip$(V4S_IP)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)pt, \
    $(V4S_PTs)))

v4s_errors_vs_pt.pdf: compare_models.py $(addprefix $(MODELS)/, $(V4S_PT_MODELS))
	$(COMPARE) --models $(V4S_PT_MODELS) --parameters $(V4S_PTs) \
    --parameter_name "Prominence Threshold" $(RPT_ARGS) $@ > v4s_errors_vs_pt.txt

# Sensitivity to IP, exponential Power for the Interval component
V4S_IPs = 0.1 0.2 0.3 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.4 0.5
V4S_IP_MODELS = $(addsuffix _masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip, \
    $(V4S_IPs)))

v4s_errors_vs_ip.pdf: compare_models.py $(addprefix $(MODELS)/, $(V4S_IP_MODELS))
	$(COMPARE) --models $(V4S_IP_MODELS) --parameters $(V4S_IPs) \
    --parameter_name "Interval Weight" \
    $@ > v4s_errors_vs_ip.txt

# Sensitivity to AutoRegressive order
V4S_ARs = 7 8 9 10 11 12 13
VAR_MODELS = $(addsuffix fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked, \
    $(addprefix varg4state_ar, $(V4S_ARs)))

v4s_errors_vs_ar.pdf: compare_models.py $(addprefix $(MODELS)/, $(VAR_MODELS))
	$(COMPARE) --models $(VAR_MODELS) --parameters $(V4S_ARs) \
--parameter_name "AR order" -- $@ > v4s_errors_vs_ar.txt

# Sensitivity to Sample Frequency (7 not allowed)
V4S_FSs = 3 4 5 6 8
VFS_MODELS = $(addsuffix lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked, \
    $(addprefix varg4state_ar$(V4S_AR)fs, $(V4S_FSs)))

v4s_errors_vs_fs.pdf: compare_models.py $(addprefix $(MODELS)/, $(VFS_MODELS))
	$(COMPARE) --models $(VFS_MODELS) --parameters $(V4S_FSs) \
--parameter_name "Sample Frequency" $@ > v4s_errors_vs_fs.txt

# Sensitivity to Low Pass Period in seconds.
V4S_LPPs = 56 57 57.5 58 58.5 59 59.5 60 60.5 61 62
LPP_MODELS = $(addsuffix rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp, \
    $(V4S_LPPs)))

v4s_errors_vs_lpp.pdf: compare_models.py $(addprefix $(MODELS)/, $(LPP_MODELS))
	$(COMPARE) --models $(LPP_MODELS) --parameters $(V4S_LPPs) \
--parameter_name "Low Pass Period" $@ > v4s_errors_vs_lpp.txt

# Exercise code
V4S_FOOs = 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65
FOO_MODELS = $(addsuffix rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp, \
    $(V4S_FOOs)))

foo.pdf: compare_models.py $(addprefix $(MODELS)/, $(FOO_MODELS))
	$(COMPARE) --n_apnea 6446 --models $(FOO_MODELS) --parameters $(V4S_FOOs) \
--parameter_name "Low Pass Period" $@ > foo.txt

# Sensitivity to Respiration Center frequency in cpm
V4S_RCs = 12.4 12.5 12.6 12.62 12.64 12.68 12.7 12.72 12.74 12.76 12.78 12.8
RC_MODELS = $(addsuffix rw$(V4S_RW)rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc, \
    $(V4S_RCs)))

v4s_errors_vs_rc.pdf: compare_models.py $(addprefix $(MODELS)/, $(RC_MODELS))
	$(COMPARE) --models $(RC_MODELS) --parameters $(V4S_RCs) \
--parameter_name "Center Frequency" $@ > v4s_errors_vs_rc.txt

# Sensitivity to Respiration width frequency in cpm
V4S_RWs = 4.8 4.81 4.82 4.83 4.84 4.85 4.852 4.854 4.856 4.858 4.86 4.865 4.87 4.875 4.88 4.89 4.9 4.91 4.92 4.94 4.96 4.98 5 5.5
RW_MODELS = $(addsuffix rs$(V4S_RS)pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw, \
    $(V4S_RWs)))

v4s_errors_vs_rw.pdf: compare_models.py $(addprefix $(MODELS)/, $(RW_MODELS))
	$(COMPARE) --models $(RW_MODELS) --parameters $(V4S_RWs) \
--parameter_name "Frequency Width" $@ > v4s_errors_vs_rw.txt

# Sensitivity to Respiration Smoothing frequency in cpm
V4S_RSs = .44 .45 .452 .454 .456 .458 .46 .462 .464
RS_MODELS = $(addsuffix pt$(V4S_PT)vp$(V4S_VP)ip$(V4S_IP)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs, \
    $(V4S_RSs)))

v4s_errors_vs_rs.pdf: compare_models.py $(addprefix $(MODELS)/, $(RS_MODELS))
	$(COMPARE) --models $(RS_MODELS) --parameters $(V4S_RSs) \
    --parameter_name "Respiration Smoothing Filter" \
    $@ > v4s_errors_vs_rs.txt

# EG $(MODELS)/varg4state_ar16fs4lpp10rc17rw4.7rs.44_init

$(MODELS)/varg4state_%_init: make_init.py model_init.py config_stats.py
	python $< four_state $* $@

$(MODELS)/varg4state%_masked: apnea_train.py $(MODELS)/varg4state%_init
	python $< --records $(APLUSNAMES) --iterations 10 $(MODELS)/varg4state$*_init $@

v4s_pass2.out: pass2.py pass1.out $(MODELS)/c_model $(V4S_BEST)
	python $< --model_paths $(MODELS)/c_model $(V4S_BEST) \
    --threshold 150.0 pass1.out temp
	cp temp $@

v4s_score.tex: score.py v4s_pass2.out
	python $^ $@  --tex

###############v2c: 2 chain model with varg, peak, interval observation ########

# Prominence Threshold
V2C_PT = 1.85
# Log Detection Threshold
V2C_LDT = 0.0
# Exponential weight for varg component
V2C_VP = 1.0
# Exponential weight for interval component
V2C_IP = 1.0

V2C_BEST = $(MODELS)/varg2chain_pt$(V2C_PT)ldt$(V2C_LDT)vp$(V2C_VP)ip$(V2C_IP)_masked

$(MODELS)/varg2chain_%_initial: make_init.py model_init.py config_stats.py
	python $< --debug --AR_order $(V2S_AR) --model_sample_frequency $(V2S_FS) \
    --low_pass_period $(V2S_LPP) --band_pass_center $(V2S_RC) --band_pass_width $(V2S_RW) \
    varg2chain $* $@


# Sensitivity to Detection Threshold
v2c_threshold.pdf: survey_power_threshold.py $(V2C_BEST)
	python $< --thresholds -1.0 1.0 21 --records $(APLUSNAMES) -- $(V2C_BEST) $@ \
    > v2c_threshold.txt

# Sensitivity to Prominence Threshold
V2C_PTs = 1.5 1.6 1.7 1.8 1.82 1.83 1.84 1.85 1.86 1.87 1.9 2 2.2 2.5 3 3.5 4 4.5 5
PT_MODELS = $(addsuffix ldt$(V2C_LDT)vp$(V2C_VP)ip$(V2C_IP)_masked, $(addprefix \
    varg2chain_pt, \
    $(V2C_PTs)))

v2c_errors_vs_pt.pdf: compare_models.py $(addprefix $(MODELS)/, $(PT_MODELS))
	$(COMPARE) --models $(PT_MODELS) --parameters $(V2C_PTs) \
    --parameter_name "Prominence Threshold" $@ > v2c_errors_vs_pt.txt

# Sensitivity to IP, exponential Power for the Interval component
V2C_IPs = .25 .5 1.0 2.0 4.0
IP_MODELS = $(addsuffix _masked, $(addprefix \
    varg2chain_pt$(V2C_PT)ldt$(V2C_LDT)vp$(V2C_VP)ip, \
    $(V2C_IPs)))

v2c_errors_vs_ip.pdf: compare_models.py $(addprefix $(MODELS)/, $(IP_MODELS))
	$(COMPARE) --models $(IP_MODELS) --parameters $(V2C_IPs) \
    --parameter_name "Interval Weight" $@ > v2c_errors_vs_ip.txt

# Sensitivity to VP, exponential Power for the VARG component
V2C_VPs = .25 .5 1.0 2.0 4.0
VP_MODELS = $(addsuffix ip$(V2C_IP)_masked, $(addprefix \
    varg2chain_pt$(V2C_PT)ldt$(V2C_LDT)vp, \
    $(V2C_VPs)))

v2c_errors_vs_vp.pdf: compare_models.py $(addprefix $(MODELS)/, $(VP_MODELS))
	$(COMPARE) --models $(VP_MODELS) --parameters $(V2C_VPs) \
    --parameter_name "VARG Weight" $@ > v2c_errors_vs_vp.txt

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
	@echo VP_MODELS: $(VP_MODELS)
	@echo V4S_BEST: $(V4S_BEST)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

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

########################Build hand_opt.pdf####################################

LIST_ERRORS = ar fs lpp rc rw rs
v4sERRORS = $(addsuffix .pdf, $(addprefix v4s_errors_vs_, $(LIST_ERRORS)))
abcdERRORS = $(addsuffix .pdf, $(addprefix abcd_, $(LIST_ERRORS)))
cheatERRORS = $(addsuffix .pdf, $(addprefix cheat_, $(LIST_ERRORS)))
v4sTHRESHOLDS = $(addsuffix .pdf, $(addprefix v4s_threshold_, a02 a09 a12 a18 a20 c10))
HANDOPT_FIGS = $(v4sERRORS) $(v4sTHRESHOLDS) $(cheatERRORS) shift_threshold.pdf v4s_threshold.pdf
hand_opt.pdf: hand_opt.tex $(HANDOPT_FIGS) v4s_score.tex abcd_score.tex
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

$(MODELS)/c_model: model_init.py
	python $< --model_sample_frequency 12 c_model $@

c_model: $(MODELS)/c_model
	ln -sf $< $@

########### 4 states with VARG and no interval observation, V4S ###########

# AutoRegressive order
V4S_AR = 5
# Model sample frequency cpm
V4S_FS = 4
# Low Pass Period seconds
V4S_LPP = 70
# Respiration Center frequency cpm
V4S_RC = 16.5
# Respiration Width cpm
V4S_RW = 3.5
# Filter for Respiration Smoothing in cpm.
V4S_RS = .49
# Detection threshold
V4S_THRESHOLD = 1.0
ABCD = --abcd -1500 1700 11 680 --shift_statistics shift_statistics.pkl
COMPARE_ABCD = python compare_models.py --records $(APLUSNAMES) $(ABCD)
COMPARE_CHEAT = python compare_models.py --records $(APLUSNAMES) --cheat

V4S_BEST = $(MODELS)/varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)_masked
# Yields 3072 errors  Earlier got 2819.  With abcd threshold function 2278.

v4s_viz.pdf: model_viz.py $(MODELS)/varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)_masked
	python $^ $@

v4s_viz.dot: model_viz.py $(MODELS)/varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)_masked
	python $^ $@

$(MODELS)/default: $(V4S_BEST)
	ln -s $< $@

default: $(V4S_BEST)
	ln -sf $< $@

write_shift_statistics.py: shift_threshold.py
	ln -s $< $@
shift_statistics.pkl: write_shift_statistics.py $(V4S_BEST)
	python $< --records $(APLUSNAMES) --  $(V4S_BEST) $@

abcd_pass2.out: pass2.py shift_statistics.pkl
	python $< --model_paths $(MODELS)/c_model $(V4S_BEST) \
    $(ABCD) pass1.out temp
	cp temp $@

abcd_score.tex: score.py abcd_pass2.out
	python $^ $@  --tex

shift_threshold.pdf: shift_threshold.py $(V4S_BEST)
	python shift_threshold.py --records $(APLUSNAMES) -- $(V4S_BEST)

threshold_statistics.pdf: threshold_statistics.py c_model default
	echo "Not implemented"
	python $^ $@ $(APLUSNAMES)

v4s_opt: optimize.py
	python $< --records  $(APLUSNAMES) -- $@

TARGS = --threshold $(V4S_THRESHOLD) $(V4S_BEST)

v4s_threshold_a01.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -2 0 21  --records a01 $(TARGS) $@

v4s_threshold_a02.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -3 -1 21  --records a02 $(TARGS) $@

v4s_threshold_a03.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds 2 4 21  --records a03 $(TARGS) $@

v4s_threshold_a07.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -1 1 21  --records a07 $(TARGS) $@


v4s_threshold_a09.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -4 -1.5 41  --records a09 $(TARGS) $@

v4s_threshold_a10.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -1 1 21  --records a10 $(TARGS) $@

v4s_threshold_a11.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -3 -1 21  --records a11 $(TARGS) $@

v4s_threshold_a12.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -3 -1 21  --records a12 $(TARGS) $@

v4s_threshold_a18.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -3 -1 21  --records a18 $(TARGS) $@

v4s_threshold_a19.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -1 1 21  --records a19 $(TARGS) $@

v4s_threshold_a20.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds -2.5 -.5 21  --records a20 $(TARGS) $@

v4s_threshold_b02.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds 0 2 21  --records b02 $(TARGS) $@

v4s_threshold_c08.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds 1.5 3.5 21  --records c08 $(TARGS) $@

v4s_threshold_c10.pdf: survey_threshold.py $(V4S_BEST)
	python $< --thresholds 1.5 3.5 21  --records c10 $(TARGS) $@

v4s_temp%.pdf: shift_threshold.py $(V4S_BEST)
	python  $< --fig_path $@ $(V4S_BEST)


COMPARE = python compare_models.py --records $(APLUSNAMES) --threshold $(V4S_THRESHOLD)

v4s_threshold.pdf: survey_threshold.py $(V4S_BEST)
	python $< --records $(APLUSNAMES) --thresholds -1.5 1.5 21 \
 $(V4S_BEST) $@ > v4s_threshold.txt

# Sensitivity to AutoRegressive order
V4S_ARs = 1 2 3 4 5 6 7 8 9 10 11
VAR_MODELS = $(addsuffix fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)_masked, $(addprefix varg4state_ar, $(V4S_ARs)))

v4s_errors_vs_ar.pdf: compare_models.py $(addprefix $(MODELS)/, $(VAR_MODELS))
	$(COMPARE) --models $(VAR_MODELS) --parameters $(V4S_ARs) \
--parameter_name "AR order" -- $@ > v4s_errors_vs_ar.txt

abcd_ar.pdf: compare_models.py $(addprefix $(MODELS)/, $(VAR_MODELS))
	$(COMPARE_ABCD) --models $(VAR_MODELS) --parameters $(V4S_ARs) \
    --parameter_name "AR order" \
    $@ > abcd_ar.txt

cheat_ar.pdf: compare_models.py $(addprefix $(MODELS)/, $(VAR_MODELS))
	$(COMPARE_CHEAT) --models $(VAR_MODELS) --parameters $(V4S_ARs) \
    --parameter_name "AR order" \
    $@ > cheat_ar.txt

# Sensitivity to Sample Frequency (7 not allowed)
V4S_FSs = 2 3 4 5 6 8 10
VFS_MODELS = $(addsuffix lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)_masked, \
    $(addprefix varg4state_ar$(V4S_AR)fs, $(V4S_FSs)))

v4s_errors_vs_fs.pdf: compare_models.py $(addprefix $(MODELS)/, $(VFS_MODELS))
	$(COMPARE) --models $(VFS_MODELS) --parameters $(V4S_FSs) \
--parameter_name "Sample Frequency" $@ > v4s_errors_vs_fs.txt

abcd_fs.pdf: compare_models.py $(addprefix $(MODELS)/, $(VFS_MODELS))
	$(COMPARE_ABCD) --models $(VFS_MODELS) --parameters $(V4S_FSs) \
    --parameter_name "Sample Frequency" \
    $@ > abcd_fs.txt

cheat_fs.pdf: compare_models.py $(addprefix $(MODELS)/, $(VFS_MODELS))
	$(COMPARE_CHEAT) --models $(VFS_MODELS) --parameters $(V4S_FSs) \
    --parameter_name "Sample Frequency" \
    $@ > cheat_fs.txt

# Sensitivity to Low Pass Period in seconds.
V4S_LPPs = 10 20 30 40 50 60 70 80
LPP_MODELS = $(addsuffix rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp, \
    $(V4S_LPPs)))

v4s_errors_vs_lpp.pdf: compare_models.py $(addprefix $(MODELS)/, $(LPP_MODELS))
	$(COMPARE) --models $(LPP_MODELS) --parameters $(V4S_LPPs) \
--parameter_name "Low Pass Period" $@ > v4s_errors_vs_lpp.txt

abcd_lpp.pdf: compare_models.py $(addprefix $(MODELS)/, $(LPP_MODELS))
	$(COMPARE_ABCD) --models $(LPP_MODELS) --parameters $(V4S_LPPs) \
    --parameter_name "Low Pass Period" \
    $@ > abcd_lpp.txt

cheat_lpp.pdf: compare_models.py $(addprefix $(MODELS)/, $(LPP_MODELS))
	$(COMPARE_CHEAT) --models $(LPP_MODELS) --parameters $(V4S_LPPs) \
    --parameter_name "Low Pass Period" \
    $@ > cheat_lpp.txt

# Exercise code
V4S_FOOs = 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65
FOO_MODELS = $(addsuffix rc$(V4S_RC)rw$(V4S_RW)rs$(V4S_RS)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp, \
    $(V4S_FOOs)))

foo.pdf: compare_models.py $(addprefix $(MODELS)/, $(FOO_MODELS))
	$(COMPARE) --n_apnea 6446 --models $(FOO_MODELS) --parameters $(V4S_FOOs) \
--parameter_name "Low Pass Period" $@ > foo.txt

# Sensitivity to Respiration Center frequency in cpm
V4S_RCs = 13 14 15 15.5 16 16.5 17 17.5 18 19 20 21
RC_MODELS = $(addsuffix rw$(V4S_RW)rs$(V4S_RS)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc, \
    $(V4S_RCs)))

v4s_errors_vs_rc.pdf: compare_models.py $(addprefix $(MODELS)/, $(RC_MODELS))
	$(COMPARE) --models $(RC_MODELS) --parameters $(V4S_RCs) \
--parameter_name "Center Frequency" $@ > v4s_errors_vs_rc.txt

abcd_rc.pdf: compare_models.py $(addprefix $(MODELS)/, $(RC_MODELS))
	$(COMPARE_ABCD) --models $(RC_MODELS) --parameters $(V4S_RCs) \
    --parameter_name "Center Frequency" \
    $@ > abcd_rc.txt

cheat_rc.pdf: compare_models.py $(addprefix $(MODELS)/, $(RC_MODELS))
	$(COMPARE_CHEAT) --models $(RC_MODELS) --parameters $(V4S_RCs) \
    --parameter_name "Center Frequency" \
    $@ > cheat_rc.txt

# Sensitivity to Respiration width frequency in cpm
V4S_RWs = 1.0 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0
RW_MODELS = $(addsuffix rs$(V4S_RS)_masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw, \
    $(V4S_RWs)))

v4s_errors_vs_rw.pdf: compare_models.py $(addprefix $(MODELS)/, $(RW_MODELS))
	$(COMPARE) --models $(RW_MODELS) --parameters $(V4S_RWs) \
--parameter_name "Frequency Width" $@ > v4s_errors_vs_rw.txt

abcd_rw.pdf: compare_models.py $(addprefix $(MODELS)/, $(RW_MODELS))
	$(COMPARE_ABCD) --models $(RW_MODELS) --parameters $(V4S_RWs) \
    --parameter_name "Frequency Width" \
    $@ > abcd_rw.txt

cheat_rw.pdf: compare_models.py $(addprefix $(MODELS)/, $(RW_MODELS))
	$(COMPARE_CHEAT) --models $(RW_MODELS) --parameters $(V4S_RWs) \
    --parameter_name "Frequency Width" \
    $@ > cheat_rw.txt

# Sensitivity to Respiration Smoothing frequency in cpm
V4S_RSs = .45 .46 .47 .48 .49 .50 .51 .52 .53 .54 .55
RS_MODELS = $(addsuffix _masked, $(addprefix \
    varg4state_ar$(V4S_AR)fs$(V4S_FS)lpp$(V4S_LPP)rc$(V4S_RC)rw$(V4S_RW)rs, \
    $(V4S_RSs)))

v4s_errors_vs_rs.pdf: compare_models.py $(addprefix $(MODELS)/, $(RS_MODELS))
	$(COMPARE) --models $(RS_MODELS) --parameters $(V4S_RSs) \
    --parameter_name "Respiration Smoothing Filter" \
    $@ > v4s_errors_vs_rs.txt

abcd_rs.pdf: compare_models.py $(addprefix $(MODELS)/, $(RS_MODELS))
	$(COMPARE_ABCD) --models $(RS_MODELS) --parameters $(V4S_RSs) \
    --parameter_name "Respiration Smoothing Filter" \
    $@ > abcd_rs.txt

cheat_rs.pdf: compare_models.py $(addprefix $(MODELS)/, $(RS_MODELS))
	$(COMPARE_CHEAT) --models $(RS_MODELS) --parameters $(V4S_RSs) \
    --parameter_name "Respiration Smoothing Filter" \
    $@ > cheat_rs.txt

# EG $(MODELS)/varg4state_ar16fs4lpp10rc17rw4.7rs.44_init

$(MODELS)/varg4state_%_init: make_init.py model_init.py
	python $< four_state $* $@

$(MODELS)/varg4state%_masked: apnea_train.py $(MODELS)/varg4state%_init
	python $< --records $(APLUSNAMES) --iterations 10 $(MODELS)/varg4state$*_init $@

v4s_pass2.out: pass2.py pass1.out $(MODELS)/c_model $(V4S_BEST)
	python $< --model_paths $(MODELS)/c_model $(V4S_BEST) \
    --threshold 150.0 pass1.out temp
	cp temp $@

v4s_score.tex: score.py v4s_pass2.out
	python $^ $@  --tex

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

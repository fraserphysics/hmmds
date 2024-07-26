######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build
DERIVED_APNEA_DATA = $(BUILD)/derived_data/apnea

# default target
heart_rate.pdf:

include Rules.mk

# In model_init.py alpha_beta = 50 10

event-2-answers:
	wget --no-check-certificate https://www.physionet.org/files/challenge-2000/1.0.0/event-2-answers

########################Build hand_opt.pdf####################################

LIST_ERRORS = ar fs lpp rc rw rs
ERRORS = $(addsuffix .pdf, $(addprefix multi_errors_vs_, $(LIST_ERRORS)))
HANDOPT_FIGS = $(ERRORS) multi_threshold.pdf
hand_opt.pdf: hand_opt.tex $(HANDOPT_FIGS) multi_score.tex test_score.tex
	pdflatex $< ; pdflatex $<

################################################################################

# FixMe: What should best_model be?

BEST_MODEL = default
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

## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py


########### MULTI: More states ###########

# AutoRegressive order
MULTI_AR = 8
# Model sample frequency cpm
MULTI_FS = 4
# Low Pass Period seconds
MULTI_LPP = 65
# Respiration Center frequency cpm
MULTI_RC = 12
# Respiration Width cpm
MULTI_RW = 3.6
# Filter for Respiration Smoothing in cpm.
MULTI_RS = .47
# Detection threshold
MULTI_THRESHOLD = 0.7

%_model_viz.pdf: model_viz.py $(MODELS)/%_classless
	python $^ $@
####################################################################

$(MODELS)/multi_%_init: make_init.py model_init.py
	python $< multi_state $* $@

MULTI_BEST = $(MODELS)/multi_ar$(MULTI_AR)fs$(MULTI_FS)lpp$(MULTI_LPP)rc$(MULTI_RC)rw$(MULTI_RW)rs$(MULTI_RS)_masked


$(MODELS)/default: $(MULTI_BEST)
	ln -sf $< $@

default: $(MULTI_BEST)
	ln -sf $< $@

$(MODELS)/multi%_masked: apnea_train.py $(MODELS)/multi%_init
	python $< --records $(TRAIN_NAMES) --iterations 40 $(MODELS)/multi$*_init $@

multi_viz.pdf: model_viz.py $(MULTI_BEST)
	python $^ $@

COMPARE = python compare_models.py --records $(TRAIN_NAMES) --threshold $(MULTI_THRESHOLD)

multi_threshold.pdf: survey_threshold.py $(MULTI_BEST)
	python $< --records $(TRAIN_NAMES) --thresholds -0.3 0.3 21 \
 $(MULTI_BEST) $@ > multi_threshold.txt

# Sensitivity to AutoRegressive order
MULTI_ARs = 5 6 7 8 9 10 11 12
VAR_MULTI = $(addsuffix fs$(MULTI_FS)lpp$(MULTI_LPP)rc$(MULTI_RC)rw$(MULTI_RW)rs$(MULTI_RS)_masked, $(addprefix multi_ar, $(MULTI_ARs)))

multi_errors_vs_ar.pdf: compare_models.py $(addprefix $(MODELS)/, $(VAR_MULTI))
	$(COMPARE) --models $(VAR_MULTI) --parameters $(MULTI_ARs) \
--parameter_name "AR order" -- $@ > multi_errors_vs_ar.txt

# Sensitivity to Sample Frequency (7 not allowed)
MULTI_FSs = 2 3 4 5 6 8
VFS_MULTI = $(addsuffix lpp$(MULTI_LPP)rc$(MULTI_RC)rw$(MULTI_RW)rs$(MULTI_RS)_masked, \
    $(addprefix multi_ar$(MULTI_AR)fs, $(MULTI_FSs)))

multi_errors_vs_fs.pdf: compare_models.py $(addprefix $(MODELS)/, $(VFS_MULTI))
	$(COMPARE) --models $(VFS_MULTI) --parameters $(MULTI_FSs) \
--parameter_name "Sample Frequency" $@ > multi_errors_vs_fs.txt

# Sensitivity to Low Pass Period in seconds.
MULTI_LPPs = 55 60 65 68 70 72 75 85
LPP_MULTI = $(addsuffix rc$(MULTI_RC)rw$(MULTI_RW)rs$(MULTI_RS)_masked, $(addprefix \
    multi_ar$(MULTI_AR)fs$(MULTI_FS)lpp, \
    $(MULTI_LPPs)))

multi_errors_vs_lpp.pdf: compare_models.py $(addprefix $(MODELS)/, $(LPP_MULTI))
	$(COMPARE) --models $(LPP_MULTI) --parameters $(MULTI_LPPs) \
--parameter_name "Low Pass Period" $@ > multi_errors_vs_lpp.txt

# Sensitivity to Respiration Center frequency in cpm
MULTI_RCs = 11 11.5 11.8 12 12.2 12.5 13 14
RC_MULTI = $(addsuffix rw$(MULTI_RW)rs$(MULTI_RS)_masked, $(addprefix \
    multi_ar$(MULTI_AR)fs$(MULTI_FS)lpp$(MULTI_LPP)rc, \
    $(MULTI_RCs)))

multi_errors_vs_rc.pdf: compare_models.py $(addprefix $(MODELS)/, $(RC_MULTI))
	$(COMPARE) --models $(RC_MULTI) --parameters $(MULTI_RCs) \
--parameter_name "Center Frequency" $@ > multi_errors_vs_rc.txt

# Sensitivity to Respiration width frequency in cpm
MULTI_RWs = 3.3 3.4 3.5 3.6 3.7 4.0
RW_MULTI = $(addsuffix rs$(MULTI_RS)_masked, $(addprefix \
    multi_ar$(MULTI_AR)fs$(MULTI_FS)lpp$(MULTI_LPP)rc$(MULTI_RC)rw, \
    $(MULTI_RWs)))

multi_errors_vs_rw.pdf: compare_models.py $(addprefix $(MODELS)/, $(RW_MULTI))
	$(COMPARE) --models $(RW_MULTI) --parameters $(MULTI_RWs) \
--parameter_name "Frequency Width" $@ > multi_errors_vs_rw.txt

# Sensitivity to Respiration Smoothing frequency in cpm
MULTI_RSs = .46 .465 .47 .475 .48
RS_MULTI = $(addsuffix _masked, $(addprefix \
    multi_ar$(MULTI_AR)fs$(MULTI_FS)lpp$(MULTI_LPP)rc$(MULTI_RC)rw$(MULTI_RW)rs, \
    $(MULTI_RSs)))

multi_errors_vs_rs.pdf: compare_models.py $(addprefix $(MODELS)/, $(RS_MULTI))
	$(COMPARE) --models $(RS_MULTI) --parameters $(MULTI_RSs) \
    --parameter_name "Respiration Smoothing Filter" \
    $@ > multi_errors_vs_rs.txt

viterbi_pass2.out: pass2.py pass1.out $(MODELS)/c_model $(MULTI_BEST)
	python $< --viterbi --model_paths $(MODELS)/c_model $(MULTI_BEST) pass1.out temp
	cp temp $@

multi_pass2.out: pass2.py $(MULTI_BEST)
	python $^ temp --records $(TRAIN_NAMES) --threshold 0.7
	cp temp $@

test_pass2.out: pass2.py $(MULTI_BEST)
	python $^ temp --records $(XNAMES) --threshold 0.7
	cp temp $@

%score.tex: score.py %pass2.out
	python $^ $@  --tex

test_score.tex: score.py test_pass2.out
	python $^ $@ $(XNAMES) --expert src/hmmds/applications/apnea/event-2-answers --tex

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
	@echo VAR_MULTI: $(VAR_MULTI)
	@echo MULTI_ARs: $(MULTI_ARs)
	@echo MULTI_BEST: $(MULTI_BEST)
## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

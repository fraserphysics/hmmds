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
ERRORS = $(addsuffix .pdf, $(addprefix errors_vs_, $(LIST_ERRORS)))
HANDOPT_FIGS = $(ERRORS) threshold.pdf viz.pdf
hand_opt.pdf: hand_opt.tex $(HANDOPT_FIGS) score.tex test_score.tex
	pdflatex $< ; pdflatex $<

################################################################################

# Lengths of time series for each training record
lengths.txt: test_alignment.py $(BEST)
	python $< --model $(BEST) > $@


## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py


########### MULTI: More states ###########

# AutoRegressive order
AR = 8
# Model sample frequency cpm
FS = 4
# Low Pass Period seconds
LPP = 65
# Respiration Center frequency cpm
RC = 12
# Respiration Width cpm
RW = 3.6
# Filter for Respiration Smoothing in cpm.
RS = .47
# Detection threshold
THRESHOLD = 0.7

####################################################################

$(MODELS)/%_init: make_init.py model_init.py
	mkdir -p  $(@D)
	python $< multi_state $* $@

BEST = $(MODELS)/ar$(AR)fs$(FS)lpp$(LPP)rc$(RC)rw$(RW)rs$(RS)_masked
$(MODELS)/default: $(BEST)
	ln -s $< $@

$(MODELS)/%_masked: apnea_train.py $(MODELS)/%_init
	python $< --records $(TRAIN_NAMES) --iterations 40 $(MODELS)/$*_init $@

viz.pdf: model_viz.py $(BEST)
	python $^ $@

COMPARE = python compare_models.py --records $(TRAIN_NAMES) --threshold $(THRESHOLD)

threshold.pdf: survey_threshold.py $(BEST)
	python $< --records $(TRAIN_NAMES) --thresholds -0.3 0.3 21 \
 $(BEST) $@ > threshold.txt

# Sensitivity to AutoRegressive order
ARs = 5 6 7 8 9 10 11 12
AR_MODELS = $(addsuffix fs$(FS)lpp$(LPP)rc$(RC)rw$(RW)rs$(RS)_masked, $(addprefix ar, $(ARs)))

errors_vs_ar.pdf: compare_models.py $(addprefix $(MODELS)/, $(AR_MODELS))
	$(COMPARE) --models $(AR_MODELS) --parameters $(ARs) \
--parameter_name "AR order" -- $@ > errors_vs_ar.txt

# Sensitivity to Sample Frequency (7 not allowed)
FSs = 2 3 4 5 6 8
FS_MODELS = $(addsuffix lpp$(LPP)rc$(RC)rw$(RW)rs$(RS)_masked, \
    $(addprefix ar$(AR)fs, $(FSs)))

errors_vs_fs.pdf: compare_models.py $(addprefix $(MODELS)/, $(FS_MODELS))
	$(COMPARE) --models $(FS_MODELS) --parameters $(FSs) \
--parameter_name "Sample Frequency" $@ > errors_vs_fs.txt

# Sensitivity to Low Pass Period in seconds.
LPPs = 55 60 65 68 70 72 75 85
LPP_MODELS = $(addsuffix rc$(RC)rw$(RW)rs$(RS)_masked, $(addprefix \
    ar$(AR)fs$(FS)lpp, \
    $(LPPs)))

errors_vs_lpp.pdf: compare_models.py $(addprefix $(MODELS)/, $(LPP_MODELS))
	$(COMPARE) --models $(LPP_MODELS) --parameters $(LPPs) \
--parameter_name "Low Pass Period" $@ > errors_vs_lpp.txt

# Sensitivity to Respiration Center frequency in cpm
RCs = 11 11.5 11.8 12 12.2 12.5 13 14
RC_MODELS = $(addsuffix rw$(RW)rs$(RS)_masked, $(addprefix \
    ar$(AR)fs$(FS)lpp$(LPP)rc, \
    $(RCs)))

errors_vs_rc.pdf: compare_models.py $(addprefix $(MODELS)/, $(RC_MODELS))
	$(COMPARE) --models $(RC_MODELS) --parameters $(RCs) \
--parameter_name "Center Frequency" $@ > errors_vs_rc.txt

# Sensitivity to Respiration width frequency in cpm
RWs = 3.3 3.4 3.5 3.6 3.7 4.0
RW_MODELS = $(addsuffix rs$(RS)_masked, $(addprefix \
    ar$(AR)fs$(FS)lpp$(LPP)rc$(RC)rw, \
    $(RWs)))

errors_vs_rw.pdf: compare_models.py $(addprefix $(MODELS)/, $(RW_MODELS))
	$(COMPARE) --models $(RW_MODELS) --parameters $(RWs) \
--parameter_name "Frequency Width" $@ > errors_vs_rw.txt

# Sensitivity to Respiration Smoothing frequency in cpm
RSs = .46 .465 .47 .475 .48
RS_MODELS = $(addsuffix _masked, $(addprefix \
    ar$(AR)fs$(FS)lpp$(LPP)rc$(RC)rw$(RW)rs, \
    $(RSs)))

errors_vs_rs.pdf: compare_models.py $(addprefix $(MODELS)/, $(RS_MODELS))
	$(COMPARE) --models $(RS_MODELS) --parameters $(RSs) \
    --parameter_name "Respiration Smoothing Filter" \
    $@ > errors_vs_rs.txt

pass2.out: pass2.py $(BEST)
	python $^ temp --records $(TRAIN_NAMES) --threshold 0.7
	cp temp $@

test_pass2.out: pass2.py $(BEST)
	python $^ temp --records $(XNAMES) --threshold 0.7
	cp temp $@

score.tex: score.py pass2.out
	python $^ $@  --tex

test_score.tex: score.py test_pass2.out
	python $^ $@ $(XNAMES) --expert raw_data/apnea/event-2-answers --tex

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
	@echo ARs: $(ARs)
	@echo BEST: $(BEST)
## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

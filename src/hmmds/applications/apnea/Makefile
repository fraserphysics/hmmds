######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
RAW = $(ROOT)/raw_data/apnea/apnea-ecg-database-1.0.0
DERIVED = $(ROOT)/build/derived_data/apnea
EXPERT =  $(ROOT)/raw_data/apnea/summary_of_training

# I made the Rtimes files using my project wfdb which imports
# PhysioNet's wfdb using its own shell.nix that is incompatible with
# qt.  On the basis of visually comparing spectrograms, I think the
# Rtimes files in hmmds3 are better.  If I find/build a better beat
# detector, perhaps I should shorten the window for calculating
# deviations.
RTIMES = $(DERIVED)/Rtimes
LPHR = $(DERIVED)/Lphr
RESPIRE = $(DERIVED)/Respire


#####################Section derived from my wfdb Makefile##################
XNAMES = x01 x02 x03 x04 x05 x06 x07 x08 x09 x10 \
x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 \
x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 \
x31 x32 x33 x34 x35

ANAMES = a01 a02 a03 a04 a05 a06 a07 a08 a09 a10 \
a11 a12 a13 a14 a15 a16 a17 a18 a19 a20

BNAMES = b01 b02 b03 b04 b05

CNAMES = c01 c02 c03 c04 c05 c06 c07 c08 c09 c10

NAMES = $(ANAMES) $(BNAMES) $(CNAMES) $(XNAMES)

$(RESPIRE)/flag: respire.py $(EXPERT) $(RTIMES)/flag
	mkdir -p $(RESPIRE)
	python respire.py $(EXPERT) $(RTIMES) $(RESPIRE)
	touch $@

$(LPHR)/flag : $(addsuffix .lphr, $(addprefix $(LPHR)/, $(NAMES)))
	touch $@

# Pattern rule for making $(LPHR)/foo.lphr from $(RTIMES)/foo.rtimes
$(LPHR)/%.lphr: rtimes2hr.py $(RTIMES)/%.rtimes
	mkdir -p $(LPHR)
	python $< $(RTIMES) $(LPHR) $*

#######################End of Copied Section ##############################

sgram.pdf: spectrogram.py
	python spectrogram.py --time_window 40 220  --frequency_window 1 30 --name a11 \
$(LPHR) $(RESPIRE) $(EXPERT) $@

# HMMs go in ${MODELS}
MODELS = ${ROOT}/build/derived_data/apnea/models
EXPERT = ${ROOT}/raw_data/apnea/summary_of_training
DATA = ${ROOT}/build/derived_data/apnea

#$(MODELS)/model_A2:

${DATA}/score_report:

.PHONY : trained
trained: ${MODELS}/model_Low ${MODELS}/model_Medium ${MODELS}/model_High

## pass1_report                      : Classify records High Medium or Low
.PHONY : pass1_report
pass1_report: ${DATA}/pass1_report.pickle

As = a01 a03 a05 a07 a09 a11 a13 a15 a17 a19 a02 a04 a06 a08 a10 a12	\
a14 a16 a18 a20
Bs = b01  b02  b03  b04
Cs = c01  c02  c03  c04  c05  c06  c07  c08  c09  c10
Xs = x01 x04 x07 x10 x13 x16 x19 x22 x25 x28 x31 x34 x02 x05 x08 x11	\
x14 x17 x20 x23 x26 x29 x32 x35 x03 x06 x09 x12 x15 x18 x21 x24 x27	\
x30 x33

# Rule for initial models: initial_A2, initial_C1, initial_High, initial_Medium, initial_Low
${MODELS}/initial_%: model_init.py utilities.py observation.py
	mkdir -p ${MODELS}
	python model_init.py --root ${ROOT} $* $@

# The trained files are expensive to build.  Don't delete them
.PRECIOUS: $(addprefix ${MODELS}/, modelA2 model_Low model_Medium model_High initial_Low)

${MODELS}/model_A2: apnea_train.py ${MODELS}/initial_A2
	python apnea_train.py --root ${ROOT} A2 ${MODELS}/initial_A2 $@

# The model initial_C1 has only one state and does not require training
${MODELS}/model_C1: ${MODELS}/initial_C1
	cp $< $@

# Use model_A2 and model_C1 to create file with lines like: x24 # Low
# stat= 1.454 llr= -0.603 R= 1.755.  For each line, calculate
# Low/Medium/High using stat, low_stat, and high_stat.
${DATA}/pass1_report.pickle: pass1.py ${MODELS}/model_A2 ${MODELS}/model_C1
	python pass1.py --root ${ROOT}

# Rule to make trained models (model_High, model_Medium, model_Low).
# Make maps % -> $* which apnea_train uses to select training data
# based on the classifications Low/Medium/High in pass1_report
${MODELS}/model_%: apnea_train.py ${MODELS}/initial_% ${DATA}/pass1_report.pickle
	python apnea_train.py --root ${ROOT} $* ${MODELS}/initial_$* $@

LMH_MODELS = $(addprefix ${MODELS}/, model_Low model_Medium model_High)
# Classify each minute.
${DATA}/pass2_report: pass2.py ${DATA}/pass1_report ${LMH_MODELS} develop.py
	python pass2.py --names ${As} ${Bs} ${Cs}  --root ${ROOT} $@

## score_report                   : Text report comparing classifications by minute
.PHONY : score_report
score_report: ${DATA}/score_report
	cp $< $@
$(DATA)/score_report: score.py ${DATA}/pass2_report
	python score.py --root $(ROOT) $(DATA)/pass2_report ${EXPERT} $@ $(As) $(Bs) $(Cs)

## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo LMH_MODLES: $(LMH_MODELS)
	@echo MODELS: $(MODELS)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

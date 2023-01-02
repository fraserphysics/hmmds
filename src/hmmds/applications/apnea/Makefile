######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

ApneaDerivedData = $(ROOT)/build/derived_data/apnea
# Default target
$(ApneaDerivedData)/pass1_report.pickle:

include Rules.mk
# Rules.mk defines RAW_DATA EXPERT LPHR RESPIRE RTIMES MODELS

${DATA}/score_report:

.PHONY : trained
trained: ${MODELS}/model_Low ${MODELS}/model_Medium ${MODELS}/model_High

As = a01 a03 a05 a07 a09 a11 a13 a15 a17 a19 a02 a04 a06 a08 a10 a12	\
a14 a16 a18 a20
Bs = b01  b02  b03  b04
Cs = c01  c02  c03  c04  c05  c06  c07  c08  c09  c10
Xs = x01 x04 x07 x10 x13 x16 x19 x22 x25 x28 x31 x34 x02 x05 x08 x11	\
x14 x17 x20 x23 x26 x29 x32 x35 x03 x06 x09 x12 x15 x18 x21 x24 x27	\
x30 x33

# ToDo: Move much of the stuff below to Rules.mk

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

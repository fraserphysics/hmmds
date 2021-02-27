######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript

# HMMs go in ${MODELS}
MODELS = ../../../derived_data/apnea/models
EXPERT = ../../../raw_data/apnea/summary_of_training

# sub-directories of ${DATA}: low_pass_heart_rate, respiration, models
DATA = ../../../derived_data/apnea

## pass1_report                      : Classify records High Medium or Low
.PHONY : pass1_report.pickle
pass1_report: ${DATA}/pass1_report.pickle

As = a01 a03 a05 a07 a09 a11 a13 a15 a17 a19 a02 a04 a06 a08 a10 a12	\
a14 a16 a18 a20
Bs = b01  b02  b03  b04
Cs = c01  c02  c03  c04  c05  c06  c07  c08  c09  c10
Xs = x01 x04 x07 x10 x13 x16 x19 x22 x25 x28 x31 x34 x02 x05 x08 x11	\
x14 x17 x20 x23 x26 x29 x32 x35 x03 x06 x09 x12 x15 x18 x21 x24 x27	\
x30 x33

COMMON_ARGS = \
--heart_rate ../../../derived_data/apnea/low_pass_heart_rate \
--respiration ../../../derived_data/apnea/respiration \
--models ${MODELS} \
--data ${DATA} \
--iterations 20 \
--pass1 ../../../derived_data/apnea/pass1_report \
--expert ../../../raw_data/apnea/summary_of_training \
--low_line 1.82 \
--high_line 2.60 \
--stat_slope 0.5 \
--Amodel model_A2 \
--BCmodel model_C1

# Rule for initial models: initial_A2, initial_C1, initial_High, initial_Medium, initial_Low
${MODELS}/initial_%: model_init.py
	python model_init.py  ${COMMON_ARGS} $* $@

${MODELS}/model_A2: ${MODELS}/initial_A2
	python apnea_train.py ${COMMON_ARGS} A2 ${MODELS}/initial_A2 $@

# The model initial_C1 has only one state and does not require training
${MODELS}/model_C1: ${MODELS}/initial_C1
	cp $< $@

# Use model_A2 and model_C1 to create file with lines like: x24 # Low
# stat= 1.454 llr= -0.603 R= 1.755.  For each line, calculate
# Low/Medium/High using stat, low_stat, and high_stat.
${DATA}/pass1_report.pickle: pass1.py ${MODELS}/model_A2 ${MODELS}/model_C1
	python pass1.py ${COMMON_ARGS}

# Rule to make trained models: model_High, model_Medium, model_Low.  Make
# maps % -> $* which apnea_train combines with the classifications
# Low/Medium/High in pass1_report to select data for training
${MODELS}/model_%:
	python apnea_train.py ${COMMON_ARGS} $* ${MODELS}/initial_$* $@

LMH_MODELS = $(addprefix ${MODLES}/, model_Low model_Medium model_High)
# Classify each minute.
${DATA}/pass2_report: pass2.py ${DATA}/pass1_report ${LMH_MODELS}
	python pass2.py ${COMMON_ARGS} --records ${As} ${Bs} ${Cs} ${Xs} --models ${LMH_MODELS} $@

## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

######################Apnea#########################################
# See ~/projects/hmmds3/code/applications/apnea/SConscript
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../..
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

# default target
APNEA_DEFAULT = $(BUILD)/TeX/apnea/hand_opt.pdf
.PHONY : evince
evince: $(APNEA_DEFAULT)
	evince $< &

include Rules.mk

# explore.py needs $(MODELS)/default
$(MODELS)/default: $(BEST)
	ln -s $< $@

event-2-answers:
	wget --no-check-certificate https://www.physionet.org/files/challenge-2000/1.0.0/event-2-answers

# Lengths of time series for each training record
lengths.txt: test_alignment.py $(BEST)
	python $< --model $(BEST) > $@

## test                           : Run tests defined in test.py
.PHONY : test
test:
	py.test test.py

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

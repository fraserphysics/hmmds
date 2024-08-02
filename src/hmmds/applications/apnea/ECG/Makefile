######################ECG#########################################
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../../..
ABSROOT = $(abspath $(ROOT))
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

# Default target.  Note: ecg.pdf depends indirectly on
# $(ECG_DERIVED)/all_selves, and building $(ECG_DERIVED)/all_selves
# builds the heart rate files that I use for apnea detection.
ecg.pdf:

include Rules.mk

elgendi.pdf: elgendi.py
	python $< $@

constant_a03.pdf: constant_a03.py  $(RTIMES)a03.ecg
	python $^ $@

a03_states_56.pdf:  ecg_states_fig.py $(ECG_DERIVED)/a03_self_AR3/states
	python $< $(RTIMES)/a03.ecg $(ECG_DERIVED)/a03_self_AR3/states 56.25 56.4 $@

a01c02_states.pdf:  a01c02_states.py $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02
	python $< $(RTIMES)  $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02 300.0 300.05 $@

self_table.tex: self_table.py $(ECG_DERIVED)/all_selves
	python self_table.py $(ECG_DERIVED) $@

table.tex: table.py $(ECG_DERIVED)/a01_trained_AR3/all_states_likelihood_heart_rate
	python table.py $(ECG_DERIVED)/a01_trained_AR3/ $@

%.pdf : %.fig
	fig2dev -L pdftex -F $< $@
%.pdf_t: %.fig %.pdf
	fig2dev -L pdftex_t -p $*.pdf $< $@

DS23FIGS = elgendi.pdf constant_a03.pdf a03_states_56.pdf a01c02_states.pdf ecg_hmm.pdf ecg_hmm.pdf_t simulated.pdf

ds23.pdf: ds23.tex $(DS23FIGS)
	export TEXINPUTS=$(ABSROOT)//:; pdflatex ds23.tex; pdflatex ds23.tex

As = a01 a03 a05 a07 a09 a11 a13 a15 a17 a19 a02 a04 a06 a08 a10 a12	\
a14 a16 a18 a20
Bs = b01  b02  b03  b04
Cs = c01  c02  c03  c04  c05  c06  c07  c08  c09  c10
Xs = x01 x04 x07 x10 x13 x16 x19 x22 x25 x28 x31 x34 x02 x05 x08 x11	\
x14 x17 x20 x23 x26 x29 x32 x35 x03 x06 x09 x12 x15 x18 x21 x24 x27	\
x30 x33

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo ECG_FIGS: $(ECG_FIGS)
	@echo ECG_DERIVED: $(ECG_DERIVED)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

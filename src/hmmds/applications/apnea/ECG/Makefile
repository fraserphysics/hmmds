######################ECG#########################################
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../../..
ABSROOT = $(abspath $(ROOT))
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

# Default target
ds23.pdf:
ecg.pdf:

include Rules.mk

elgendi.pdf: elgendi.py
	python $< $@

constant_a03.pdf: constant_a03.py  $(RTIMES)a03.ecg
	python $^ $@

a03_states_56.pdf:  ecg_states_fig.py $(ECG)/a03_self_AR3/states
	python ecg_states_fig.py $(RTIMES)/a03.ecg $(ECG)/a03_self_AR3/states 56.25 56.4 $@

a01c02_states.pdf:  a01c02_states.py $(ECG)/a01_trained_AR3/states/a01 $(ECG)/a01_trained_AR3/states/c02
	python $< $(RTIMES)  $(ECG)/a01_trained_AR3/states/a01 $(ECG)/a01_trained_AR3/states/c02 300.0 300.05 $@

a01a19c02.pdf:  a01a19c02.py $(ECG)/a01_trained_AR3/states/a01 $(ECG)/a01_trained_AR3/states/c02
	python $< $(RTIMES) $(ECG)/a01_trained_AR3/states/c02 300.0 300.05 $@

%_states_71.pdf: ecg_states_fig.py $(ECG)/%/states/a01
	python ecg_states_fig.py $(RTIMES)/a01.ecg $(ECG)/$*/states/a01 71.1 71.14 $@

%_states_70.pdf: ecg_states_fig.py $(ECG)/%/states/a01
	python ecg_states_fig.py $(RTIMES)/a01.ecg $(ECG)/$*/states/a01 70.785 70.825 $@

train_log.pdf: train_characteristic.py $(ECG)/a01_trained_AR3/unmasked_trained
	python $< $(ECG)/a01_trained_AR3/unmasked_trained.log $@

self_table.tex: self_table.py $(ECG)/all_selves
	python self_table.py $(ECG) $@

table.tex: table.py $(ECG)/a01_trained_AR3/all_states_likelihood_heart_rate
	python table.py $(ECG)/a01_trained_AR3/ $@

like_a14_x07.pdf: plot_like.py $(ECG)/a01_trained_AR3/all_states_likelihood_heart_rate
	python plot_like.py $(RTIMES) $(ECG)/a01_trained_AR3/likelihood/ 300.16 300.19 a14 x07 $@

simulated.pdf: plot_simulation.py $(ECG)/a01_trained_AR3/unmasked_trained
	python $^ 1000 $@

%.pdf : %.fig
	fig2dev -L pdftex -F $< $@
%.pdf_t: %.fig %.pdf
	fig2dev -L pdftex_t -p $*.pdf $< $@

ECG_FIGS = a01_trained_AR3_states_70.pdf a01_trained_AR3_states_71.pdf a01a19c02.pdf train_log.pdf like_a14_x07.pdf simulated.pdf

DS23FIGS = elgendi.pdf constant_a03.pdf a03_states_56.pdf a01c02_states.pdf ecg_hmm.pdf ecg_hmm.pdf_t simulated.pdf

ds23.pdf: ds23.tex $(DS23FIGS)
	export TEXINPUTS=$(ABSROOT)//:; pdflatex ds23.tex; pdflatex ds23.tex

ecg.pdf: ecg.tex $(ECG_FIGS) table.tex self_table.tex
	pdflatex $< ; pdflatex $<

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

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

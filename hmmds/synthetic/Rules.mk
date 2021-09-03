# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the root of this project and
# BUILD is where derived results go.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
SYNTHETIC_CODE = $(ROOT)/hmmds/synthetic
# SYNTHETIC_CODE is this directory

$(SYNTHETIC_DATA)/lorenz.flag: $(SYNTHETIC_CODE)/lorenz.py
	mkdir -p $(SYNTHETIC_DATA)/TSintro
	python $< --n_samples 2050 --quantfile $(SYNTHETIC_DATA)/lorenz.4 --xyzfile $(SYNTHETIC_DATA)/lorenz.xyz  --TSintro $(SYNTHETIC_DATA)/TSintro
	touch $@

$(SYNTHETIC_DATA)/states: $(SYNTHETIC_CODE)/StatePic.py $(SYNTHETIC_DATA)/m12s.4y
	python $<  $(SYNTHETIC_DATA) lorenz.4 lorenz.xyz m12s.4y

$(SYNTHETIC_DATA)/m12s.4y : $(SYNTHETIC_CODE)/MakeModel.py $(SYNTHETIC_DATA)/lorenz.xyz
	python $(SYNTHETIC_CODE)/MakeModel.py ${N_TRAIN} $(SYNTHETIC_DATA) lorenz.4 m12s.4y

$(SYNTHETIC_DATA)/lorenz.xyz: $(SYNTHETIC_CODE)/lorenz.py
	python $< --n_samples 20000 --levels 4 --quantfile $(SYNTHETIC_DATA)/lorenz.4 --xyzfile $@

# Local Variables:
# mode: makefile
# End:

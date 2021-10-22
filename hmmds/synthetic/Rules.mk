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

# vstates is a sentinel for varg_stateN (N in 0...11)  ToDo: implement this
$(SYNTHETIC_DATA)/vstates: $(SYNTHETIC_CODE)/VStatePic.py $(SYNTHETIC_DATA)/lorenz.flag
	python $<  --data_dir $(SYNTHETIC_DATA) --data_in lorenz.xyz --out_preface varg_state
	touch $@

$(SYNTHETIC_DATA)/m12s.4y : $(SYNTHETIC_CODE)/MakeModel.py $(SYNTHETIC_DATA)/lorenz.xyz
	python $(SYNTHETIC_CODE)/MakeModel.py ${N_TRAIN} $(SYNTHETIC_DATA) lorenz.4 m12s.4y

$(SYNTHETIC_DATA)/lorenz.xyz: $(SYNTHETIC_CODE)/lorenz.py
	python $< --n_samples 20000 --levels 4 --quantfile $(SYNTHETIC_DATA)/lorenz.4 --xyzfile $@

$(SYNTHETIC_DATA)/TrainChar: $(SYNTHETIC_CODE)/TrainChar.py  $(SYNTHETIC_DATA)/lorenz.flag
	python $< $(SYNTHETIC_DATA)/lorenz.4 $@

$(SYNTHETIC_DATA)/em.pickle: $(SYNTHETIC_CODE)/em.py
	python $< $@

# Sentinel for SGO_sim and SGO_train
$(SYNTHETIC_DATA)/SGO: $(SYNTHETIC_CODE)/ScalarGaussian.py
	python $< $(SYNTHETIC_DATA)
	touch $@

# Local Variables:
# mode: makefile
# End:

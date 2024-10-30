# Rules.mk: This file can be included by a makefile anywhere as long
# as HMMDS and BUILD are defined.  HMMDS is the root of this project and
# BUILD is where derived results go.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
SYNTHETIC_CODE = $(HMMDS)/synthetic
# SYNTHETIC_CODE is this directory

NLorenzBig = 40000
LorenzTrainingIterations = 100

$(SYNTHETIC_DATA)/lorenz.flag: $(SYNTHETIC_CODE)/lorenz.py
	mkdir -p $(SYNTHETIC_DATA)/TSintro
	python $< --n_samples 2050 --quantfile $(SYNTHETIC_DATA)/lorenz.4 --xyzfile $(SYNTHETIC_DATA)/lorenz.xyz  --TSintro $(SYNTHETIC_DATA)/TSintro
	touch $@

$(SYNTHETIC_DATA)/states: $(SYNTHETIC_CODE)/state_pic.py $(SYNTHETIC_DATA)/m12s.4y
	python $<  $(SYNTHETIC_DATA) lorenz.4 lorenz.xyz m12s.4y

# vstates is a sentinel for varg_stateN (N in 0...11)  ToDo: implement this
$(SYNTHETIC_DATA)/vstates: $(SYNTHETIC_CODE)/v_state_pic.py $(SYNTHETIC_DATA)/lorenz.flag
	mkdir -p $(@D)
	python $< --random_seed 7 $(SYNTHETIC_DATA) lorenz.xyz varg_state
	touch $@

$(SYNTHETIC_DATA)/m12s.4y : $(SYNTHETIC_CODE)/make_model.py $(SYNTHETIC_DATA)/lorenz.xyz
	python $< ${N_TRAIN} $(SYNTHETIC_DATA) lorenz.4 m12s.4y

$(SYNTHETIC_DATA)/lorenz.xyz: $(SYNTHETIC_CODE)/lorenz.py
	python $< --n_samples $(NLorenzBig) --levels 4 --quantfile $(SYNTHETIC_DATA)/lorenz.4 --xyzfile $@

# This takes about 11 minutes
$(SYNTHETIC_DATA)/TrainChar: $(SYNTHETIC_CODE)/train_char.py  $(SYNTHETIC_DATA)/lorenz.flag
	python $< --n_iterations=$(LorenzTrainingIterations) $(SYNTHETIC_DATA)/lorenz.4 $@

$(SYNTHETIC_DATA)/gauss_mix.pkl: $(SYNTHETIC_CODE)/gauss_mix.py
	mkdir -p $(@D)
	python $< --random_seed 10 $(@D)

# Sentinel for SGO_sim and SGO_train
$(SYNTHETIC_DATA)/SGO: $(SYNTHETIC_CODE)/scalar_gaussian.py
	mkdir -p $(@D)
	mkdir -p $(BUILD)/TeX/book/
	python $< --random_seed 5 --iterations_fail 2 $(BUILD)/TeX/book/SGO_values.tex $(SYNTHETIC_DATA)
	touch $@

# Local Variables:
# mode: makefile
# End:

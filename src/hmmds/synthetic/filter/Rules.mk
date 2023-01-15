# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the root of this project and
# BUILD is where derived results go.

FILTER_DATA = $(BUILD)/derived_data/synthetic/filter
FILTER_CODE = $(ROOT)/src/hmmds/synthetic/filter
# FILTER_CODE is this directory

FILTER_ARGS = --sample_rate 100 --sample_ratio 5 --n_fine 1000 --n_coarse 500 --a 0.01 --b 2.0 --d 25.0

$(FILTER_DATA)/%_data: $(FILTER_CODE)/%_simulation.py
	mkdir -p $(FILTER_DATA)
	python $< $(FILTER_ARGS) $@

LORENZ_ARGS = --fudge 300 --dt 0.1 --b .1 --d 2.0 --sample_ratio 5 \
--n_fine 201 --n_coarse 51 --random_seed 12

$(FILTER_DATA)/lorenz_particle_data: $(FILTER_CODE)/lorenz_particle_simulation.py  $(FILTER_CODE)/lorenz_sde.flag
	mkdir -p $(FILTER_DATA)
	python $< $(LORENZ_ARGS) $@
$(FILTER_DATA)/lorenz_data: $(FILTER_CODE)/lorenz_simulation.py  $(FILTER_CODE)/lorenz_sde.flag
	mkdir -p $(FILTER_DATA)
	python $< $(LORENZ_ARGS) $@
$(FILTER_DATA)/distribution_data: $(FILTER_CODE)/distribution.py
	mkdir -p $(FILTER_DATA)
	python $< --n_samples 10000 --a 0.05 --b 0.2 $@

$(FILTER_DATA)/log_likelihood_data: $(FILTER_CODE)/log_likelihood.py
	mkdir -p $(FILTER_DATA)
	python $<  --n_samples 1000 --n_b 10 --b_range .8 1.2 $@

# FixMe: My build structure does not get the library built
$(FILTER_CODE)/lorenz_sde.flag: $(FILTER_CODE)/lorenz_sde.pyx $(FILTER_CODE)/build.py
	cd $(FILTER_CODE) ; python build.py build_ext --inplace
	touch $@

# Local Variables:
# mode: makefile
# End:

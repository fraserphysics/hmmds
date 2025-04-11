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

$(FILTER_DATA)/particle_1k: $(FILTER_CODE)/particle.py $(FILTER_CODE)/filter.py
	mkdir -p $@
	python $< --margin 0.01 --s_augment 5e-4 --resample 10000 4000 --r_threshold 1e-3 $@

$(FILTER_DATA)/ddays: $(FILTER_CODE)/particle.py $(FILTER_CODE)/filter.py
	mkdir -p $@
	python $< --n_y 100 $@

####Structured Studies of  Effects of  Parameter on Particle Filter#############
EdgeMax = --edge_max 0.2 # Divide boxes bigger than this
HMax = --h_max 0.0001
# Runge Kutta integration time step
Margin = --margin 0.01 # Keep particles this close to boundary
NInitialize = --n_initialize 15000
NQuantized = --n_quantized 4 # Number of partition regions
NY = --n_y 5000 # Number of test observations
RExtra = --r_extra 3.0 # Extra factor for dividing boxes
RandomSeed = --random_seed 7
Resample = --resample 50000 5000 # When n particles > n_a resample to n_b
RThreshold = --r_threshold 0.0003 # Divide box when ratio of quadratic to linear
 # terms of Lorenz function this big
SAugment = --s_augment 0.001 # Amount to grow boxes at each time step
TRelax = --t_relax 50.0 # Lorenz time to relax to attractor
TimeStep = --time_step 0.15 # Lorenz time between samples

ARGS = $(EdgeMax) $(HMax) $(Margin) $(NInitialize) $(NY) $(RExtra) \
$(RandomSeed) $(Resample) $(RThreshold) $(SAugment) $(TRelax) $(TimeStep)

$(FILTER_DATA)/s_augment/%:
	mkdir -p $(@D)
	command time -o $(@D)/temp-time python $(FILTER_CODE)/wrapper_particle.py $(subst $(SAugment),--s_augment $*,$(ARGS)) $(@D)
	cd $(@D); ln -s *"s_augment..$*"* $*; cat temp-time >> $*/log.txt; rm temp-time

$(FILTER_DATA)/h_max/%:
	mkdir -p $(@D)
	command time -o $(@D)/temp-time python $(FILTER_CODE)/wrapper_particle.py $(subst $(HMax),--h_max $*,$(ARGS)) $(@D)
	cd $(@D); ln -s *"h_max..$*"* $*; cat temp-time >> $*/log.txt; rm temp-time

$(FILTER_DATA)/edge_max/%:
	mkdir -p $(@D)
	command time -o $(@D)/temp-time python $(FILTER_CODE)/wrapper_particle.py --verbose $(subst $(EdgeMax),--edge_max $*,$(ARGS)) $(@D)
	cd $(@D); ln -s *"edge_max..$*.."* $*; cat temp-time >> $*/log.txt; rm temp-time

$(FILTER_DATA)/margin/%:
	mkdir -p $(@D)
	command time -o $(@D)/temp-time python $(FILTER_CODE)/wrapper_particle.py --verbose $(subst $(Margin),--margin $*,$(ARGS)) $(@D)
	cd $(@D); ln -s *"margin..$*.."* $*; cat temp-time >> $*/log.txt; rm temp-time

$(FILTER_DATA)/r_extra/%:
	mkdir -p $(@D)
	command time -o $(@D)/temp-time python $(FILTER_CODE)/wrapper_particle.py --verbose $(subst $(RExtra),--r_extra $*,$(ARGS)) $(@D)
	cd $(@D); ln -s *"r_extra..$*.."* $*; cat temp-time >> $*/log.txt; rm temp-time

$(FILTER_DATA)/r_threshold/%:
	mkdir -p $(@D)
	command time -o $(@D)/temp-time python $(FILTER_CODE)/wrapper_particle.py --verbose $(subst $(RThreshold),--r_threshold $*,$(ARGS)) $(@D)
	cd $(@D); ln -s *"r_threshold..$*.."* $*; cat temp-time >> $*/log.txt; rm temp-time

$(FILTER_DATA)/edge_max.tex: $(FILTER_CODE)/filter_to_tex.py $(addprefix $(FILTER_DATA)/edge_max/, 1.0 0.5 0.2 0.1 0.05)
	python $^ $@

$(FILTER_DATA)/s_augment.tex: $(FILTER_CODE)/filter_to_tex.py $(addprefix $(FILTER_DATA)/s_augment/, 0.01 0.003 0.001 0.0003 0.0001)
	python $^ $@

$(FILTER_DATA)/h_max.tex: $(FILTER_CODE)/filter_to_tex.py $(addprefix $(FILTER_DATA)/h_max/, 0.003 0.001 0.0003 0.0001 3e-05 1e-05)
	python $^ $@

$(FILTER_DATA)/r_extra.tex: $(FILTER_CODE)/filter_to_tex.py $(addprefix $(FILTER_DATA)/r_extra/, 0.8 1.0 1.2 1.5 1.7 2.0 3.0 4.0 5.0)
	python $^ $@

$(FILTER_DATA)/margin.tex: $(FILTER_CODE)/filter_to_tex.py $(addprefix $(FILTER_DATA)/margin/, 0.5 0.2 0.1 0.05 0.01 0.003 0.0)
	python $^ $@

$(FILTER_DATA)/r_threshold.tex: $(FILTER_CODE)/filter_to_tex.py $(addprefix $(FILTER_DATA)/r_threshold/, 0.01 0.003 0.001 0.0003 0.0001)
	python $^ $@

# Local Variables:
# mode: makefile
# End:

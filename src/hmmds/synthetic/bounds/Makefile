ALL_FIGS =  ToyTS1.pdf ToyStretch.pdf ToyH.pdf benettin.pdf LikeLor.pdf

# Makefile: For testing scripts in this directory from this directory

ROOT = ../../../..
BUILD = $(ROOT)/build

# The default target is test
test:

# Rules.mk contains the real targets and rules to build them
include Rules.mk

# ALL is all the targets that Rules.mk is responsible for
ALL = $(addprefix $(BOUNDS_DATA)/, data_h_view toy_h.data BenettinData like_lor_data)

## test                 : Build all targets
.PHONY : test
test: $(ALL)

data_h_view: h_view.py
	@echo To make data_h_view, run \"python h_view.py\" and press the \"save\" button

toy_h.data: toy_h.py
	python $< --n_t 200 --t_steps .02 .53 .125 --log_steps -3.5 -5.6 -.5 $@

BenettinData: benettin_data.py
	python benettin_data.py --n_times=400 --n_runs=100 --n_relax 50 $@

like_lor_data: like_lor.py
	python $< $@

study10k%: particle.py filter.py
	python $< --margin 0.01 --s_augment $* --resample 10000 4000 --r_threshold 1e-5 --clouds 0 25 --n_y 25 $@

study100k%: particle.py filter.py
	python $< --margin 0.01 --s_augment $* --resample 100000 4000 --r_threshold 1e-5 --clouds 0 25 --n_y 25 $@

study1M%: particle.py filter.py
	python $< --margin 0.01 --s_augment $* --resample 1000000 4000 --r_threshold 1e-5 --clouds 0 25 --n_y 25 $@

study_long1M10k%: particle.py filter.py
	python $< --margin 0.01 --s_augment $* --resample 1000000 10000 --r_threshold 1e-5 --clouds 110 140 --n_y 140 $@


# Next block is for comparing results of parameter variations
RESAMPLE_A = 200000
RESAMPLE_B = 20000
NAME = atol..1e-08..edge_max..0.2..margin..0.01..n_initialize..15000..n_quantized..4..n_y..10..r_extra..2.0..r_threshold..THRESHOLD..random_seed..7..resample..$(RESAMPLE_A)..$(RESAMPLE_B)..s_augment..0.0005..t_relax..50.0..time_step..0.15..

Args = --r_extra 2.0 --edge_max 0.2 --margin 0.01 --s_augment 5e-4 --n_y 1000 --resample $(RESAMPLE_A) $(RESAMPLE_B)

study_threshold/%: wrapper_particle.py filter.py
	python $< $(Args) --r_threshold $* $(@D)/$(subst THRESHOLD,$*,$(NAME))
	ln -s $(*D)/$(subst THRESHOLD,$*,$(NAME)) $@

.PHONY: threshold_study
threshold_study: $(addprefix study_threshold/, 1e-2 3e-3 1e-3 3e-4 1e-4)

# python entropy_particle.py --dict_template study_threshold/{0}/dict.pkl 1e-2 1e-3 1e-4

TestArgs = --r_extra 2.0 --edge_max 0.2 --margin 0.01 --s_augment 5e-4 --n_y 20 --resample 20000 2000 --r_threshold 1e-3

.PHONY: test_args
test_args: filter.py particle.py
	rm -rf $@
	mkdir $@
	python particle.py $(TestArgs) $@


## list                 : List all files built by this makefile
.PHONY : list
list:
	ls -l $(ALL)

## clean                : Remove all files built by this makefile
.PHONY : clean
clean:
	rm -f $(ALL) *pdf*

## variables            : Print selected variables.
.PHONY : variables
variables:
	@echo FILTER_DATA: $(SYNTHETIC_CODE)

## help                 : Print comments on targets from makefile
.PHONY : help
help :
	@sed -n 's/^## / /p' Makefile

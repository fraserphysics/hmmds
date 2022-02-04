ROOT = ../../..
BUILD = $(ROOT)/build
#LinearSystemData = $(BUILD)/LinearSystemData
#LinearSystemCode = $(ROOT)/hmmds/synthetic/filter
#LinearSystemFigures = $(BUILD)/figs/LinearSystemFigures
LinearSystemData = ./
LinearSystemCode = ./
LinearSystemFigures = ./

DATA = data

SAMPLE_RATIO = 5

L_RATIO = 5

ALL_DATA = linear_map_data mimic_data linear_sde_data lorenz_data

ALL_FIGS = linear_map_filter.pdf linear_map_smooth.pdf \
linear_sde_filter.pdf linear_sde_smooth.pdf lorenz_filter.pdf \
distribution.pdf log_likelihood.pdf

LINEAR_ARGS = --sample_rate 100 --sample_ratio $(SAMPLE_RATIO) \
--n_fine 1000 --n_coarse 500 --a 0.01 --b 2.0 --d 25.0

%_data: %_simulation.py
	python $< $(LINEAR_ARGS) $@

lorenz_data: lorenz_simulation.py
	python $< --fudge 200 --dt 0.2 --b .1 --d 2.0 --sample_ratio 5 \
--n_fine 101 --n_coarse 51 --random_seed 12 $@

.PHONY: test
test:
	rm -f $(ALL_DATA) all_data
	make all_data
	python compare.py

all_data: $(ALL_DATA)
	touch $@

all_figs: $(ALL_FIGS)
	touch $@

## Stuff that belongs in Rules.mk here

distribution_data: distribution.py
	python $< --n_samples 10000 --a 0.05 --b 0.2 $@

log_likelihood_data: log_likelihood.py
	python $<  --n_samples 1000 --n_b 10 --b_range .8 1.2 $@

## Stuff that belongs in plotscripts/linearsystems

distribution.pdf: distribution_fig.py distribution_data 
	python $^ $@

log_likelihood.pdf: log_likelihood_fig.py log_likelihood_data
	python $^ $@

%_filter.pdf: %_data filter_fig.py
	python filter_fig.py --sample_ratio 5 $< $@

%_smooth.pdf: %_data smooth_fig.py
	python smooth_fig.py $< $@


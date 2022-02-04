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

NEW_FIGS = linear_map_filter.pdf linear_map_smooth.pdf linear_sde_filter.pdf linear_sde_smooth.pdf lorenz_filter.pdf

LINEAR_ARGS = --sample_rate 100 --sample_ratio $(SAMPLE_RATIO) \
--n_fine 1000 --n_coarse 500 --a 0.01 --b 2.0 --d 25.0

linear_map_data: linear_map_simulation.py
	python $< $(LINEAR_ARGS) $@

mimic_data: mimic_simulation.py
	python $< $(LINEAR_ARGS) $@

linear_sde_data: linear_sde_simulation.py
	python $< $(LINEAR_ARGS) $@

lorenz_data: lorenz_simulation.py
	python $< --b .1 --d 2.0 --sample_ratio 5 --n_fine 200 --n_coarse 101 $@

%_filter.pdf: %_data filter_fig.py
	python filter_fig.py --show --sample_ratio 5 $< $@

%_smooth.pdf: %_data smooth_fig.py
	python smooth_fig.py --show $< $@

.PHONY: test
test:
	rm -f $(ALL_DATA) all_data
	make all_data
	python compare.py
all_data: $(ALL_DATA)
	touch $@

new_figs: $(NEW_FIGS)
	touch $@

.PHONY: all_figs
all_figs: ll.pdf smooth.pdf system_and_filter.pdf distribution.pdf sde_smooth.pdf sde.pdf

ll.pdf: max_like.py
	python max_like.py --show --n_samples 1000 --n_b 10 --b_range .8 1.2 ll.pdf

# The default target
$(LinearSystemFigures)/system_and_filter.pdf: linear_filter.py $(DATA)
	python $< --show  --sample_ratio $(SAMPLE_RATIO) $(DATA) $@

## Stuff that belongs in Rules.mk here

$(DATA): $(LinearSystemCode)/linear_simulation.py
	mkdir -p $(LinearSystemData)
	python $< $(LINEAR_ARGS) $@

sde_data: sde_simulation.py
	python $< $(LINEAR_ARGS) $@

sde.pdf: sde_data linear_filter.py

## Stuff that belongs in plotscripts/linearsystems

# Writes dist_data and distribution.pdf
distribution.pdf: distribution.py
	python $< --n_samples 10000 --a 0.05 --b 0.2 --show dist_data $@

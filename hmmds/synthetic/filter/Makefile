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

ALL_DATA = ekf_data data mimic_data
.PHONY: test
test:
	rm -f $(ALL_DATA) all_data
	make all_data
	python compare.py
all_data: $(ALL_DATA)
	touch $@

.PHONY: all_figs
all_figs: ekf.pdf ll.pdf smooth.pdf system_and_filter.pdf

ekf.pdf: linear_filter.py ekf_data
	python $< --show  --sample_ratio $(L_RATIO) ekf_data $@

ekf_data: lorenz_simulation.py
	python $< --state_noise .1 --observation_noise 2.0 --coarse_dt 0.1 \
--fudge 100 --sample_ratio $(L_RATIO) --n_fine 200 --n_coarse 101 $@

ll.pdf: max_like.py
	python max_like.py --show --n_samples 1000 --n_b 10 --b_range .8 1.2 ll.pdf

# The default target
$(LinearSystemFigures)/system_and_filter.pdf: linear_filter.py $(DATA)
	python $< --show  --sample_ratio $(SAMPLE_RATIO) $(DATA) $@

## Stuff that belongs in Rules.mk here


LINEAR_ARGS = --sample_rate 100 --sample_ratio $(SAMPLE_RATIO) \
--n_fine 1000 --n_coarse 500 --a 0.01 --b 2.0 --d 25.0
$(DATA): $(LinearSystemCode)/linear_simulation.py
	mkdir -p $(LinearSystemData)
	python $< $(LINEAR_ARGS) $@

mimic_data: mimic.py
	python $< $(LINEAR_ARGS) $@

mimic.pdf: mimic_data
	python linear_filter.py --show  --sample_ratio $(SAMPLE_RATIO) $< $@

mimic_smooth.pdf: mimic_data
	python plot_smooth.py --show $< $@

## Stuff that belongs in plotscripts/linearsystems

# Writes dist_data and distribution.pdf
distribution.pdf: distribution.py
	python $< --n_samples 10000 --a 0.05 --b 0.2 --show dist_data $@

smooth.pdf: plot_smooth.py $(DATA)
	python plot_smooth.py --show $(DATA) $@

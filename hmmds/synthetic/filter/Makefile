ROOT = ../../..
BUILD = $(ROOT)/build
#LinearSystemData = $(BUILD)/LinearSystemData
#LinearSystemCode = $(ROOT)/hmmds/synthetic/filter
#LinearSystemFigures = $(BUILD)/figs/LinearSystemFigures
LinearSystemData = ./
LinearSystemCode = ./
LinearSystemFigures = ./

DATA = data

ekf.pdf: linear_filter.py ekf_data
	python $< --show  --sample_ratio $(SAMPLE_RATIO) ekf_data $@

ekf_data: lorenz_simulation.py
	python $< $@

ll.pdf: max_like.py
	python max_like.py --show --n_samples 1000 --n_b 10 --b_range .8 1.2 ll.pdf

SAMPLE_RATIO = 5
# The default target
$(LinearSystemFigures)/system_and_filter.pdf: linear_filter.py $(DATA)
	python $< --show  --sample_ratio $(SAMPLE_RATIO) $(DATA) $@

## Stuff that belongs in Rules.mk here


$(DATA): $(LinearSystemCode)/linear_simulation.py
	mkdir -p $(LinearSystemData)
	python $< --sample_rate 100 --sample_ratio $(SAMPLE_RATIO) \
--n_fine 1000 --n_coarse 500 --a 0.01 --b 0.04 --d 0.5 $@

## Stuff that belongs in plotscripts/linearsystems

distribution.pdf: distribution.py
	python $< --n_samples 10000 --a 0.05 --b 0.2 --show dist_data $@

smooth.pdf: plot_smooth.py $(DATA)
	python plot_smooth.py --show $(DATA) $@

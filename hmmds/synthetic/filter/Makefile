ROOT = ../../..
BUILD = $(ROOT)/build
#LinearSystemData = $(BUILD)/LinearSystemData
#LinearSystemCode = $(ROOT)/hmmds/synthetic/filter
#LinearSystemFigures = $(BUILD)/figs/LinearSystemFigures
LinearSystemData = ./
LinearSystemCode = ./
LinearSystemFigures = ./

DATA = data

# The default target
$(LinearSystemFigures)/system_and_filter.pdf: linear_filter.py $(DATA)
	python $< --show $(DATA) $@

## Stuff that belongs in Rules.mk here


$(DATA): $(LinearSystemCode)/linear_simulation.py
	mkdir -p $(LinearSystemData)
	python $< --sample_rate 100 --sample_ratio 10 \
--n_fine 1000 --n_coarse 500 --a 0.05 --b 0.2 --d 0.1 $@

## Stuff that belongs in plotscripts/linearsystems

distribution.pdf: distribution.py
	python $< --n_samples 10000 --a 0.05 --b 0.2 --show dist_data $@

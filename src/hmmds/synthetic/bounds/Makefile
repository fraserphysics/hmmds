ALL_FIGS =  ToyTS1.pdf ToyStretch.pdf ToyH.pdf benettin.pdf LikeLor.pdf

# ToyTS1.pdf ToyStretch.pdf ToyH.pdf benettin.pdf LikeLor.pdf
.PHONY : ALL
ALL : $(ALL_FIGS)

ToyTS1.pdf: plot.py data_h_view
	python $< --ToyTS1 $@ data_h_view

ToyStretch.pdf: plot.py data_h_view
	python $< --ToyStretch $@ data_h_view

ToyH.pdf: plot_toy_h.py toy_h.data
	python $^ $@

benettin.pdf: benettin_plot.py BenettinData
	python $^ $@

LikeLor.pdf: plot_like_lor.py like_lor_data
	python $^ $@

data_h_view: h_view.py
	@echo To make data_h_view, run \"python h_view.py\" and press the \"save\" button

toy_h.data: toy_h.py
	python $< --n_t 200 --t_steps .02 .53 .125 --log_steps -3.5 -5.6 -.5 $@

BenettinData: benettin_data.py
	python benettin_data.py --n_times=400 --n_runs=100 --n_relax 50 $@

# This takes almost 8 minutes on cathcart
like_lor_data: like_lor.py
	python $< $@

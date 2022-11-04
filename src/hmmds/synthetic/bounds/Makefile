toy_h.pdf: plot_toy_h.py toy_h.data
	python $^ $@

toy_h.data: toy_h.py
	python $< --n_t 200 --t_steps .02 .53 .125 --log_steps -3.5 -5.6 -.5 $@
data_h_view: h_view.py
	@echo To make data_h_view, run \"python h_view.py\" and press the \"save\" button
ToyStretch.pdf: plot.py data_h_view
	python $< --ToyStretch $@ data_h_view

BenettinData: benettin_data.py
	python benettin_data.py --n_times=400 --n_runs=10 --n_relax 50 $@

BenettinFigure.pdf: benettin_plot.py BenettinData
	python $^ $@

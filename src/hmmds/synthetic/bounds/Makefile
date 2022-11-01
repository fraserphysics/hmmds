toy_h.pdf: plot_toy_h.py toy_h.data
	python $^ $@

toy_h.data: toy_h.py
	python $< --n_t 200 --t_steps .02 .53 .125 --log_steps -3.5 -5.6 -.5 $@

# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

FILTER_DATA = $(BUILD)/derived_data/synthetic/filter
FIGS_FILTER = $(BUILD)/figs/filter
FilterPlotscripts = $(ROOT)/src/plotscripts/filter

$(FIGS_FILTER)/%_filter.pdf: $(FilterPlotscripts)/filter_fig.py $(FILTER_DATA)/%_data
	mkdir -p $(@D)
	python $^ --sample_ratio 5 $@

$(FIGS_FILTER)/%_smooth.pdf: $(FilterPlotscripts)/smooth_fig.py $(FILTER_DATA)/%_data
	mkdir -p $(@D)
	python $^ $@

$(FIGS_FILTER)/distribution.pdf: $(FilterPlotscripts)/distribution_fig.py $(FILTER_DATA)/distribution_data
	mkdir -p $(@D)
	python $^ $@

$(FIGS_FILTER)/log_likelihood.pdf: $(FilterPlotscripts)/log_likelihood_fig.py $(FILTER_DATA)/log_likelihood_data
	mkdir -p $(@D)
	python $^ $@

# Don't depend on the dir ddays because it's date may not be right for gnu-make
$(FIGS_FILTER)/filter_b.pdf: $(FilterPlotscripts)/forecast_update.py $(FILTER_DATA)/ddays/dict.pkl
	mkdir -p $(@D)
	python $< $(dir $(word 2, $^)) --start 72 $@

$(FIGS_FILTER)/no_divide.jpeg: $(FilterPlotscripts)/forecast_update.py $(FILTER_DATA)/ddays/dict.pkl
	mkdir -p $(@D)
	python $< $(dir $(word 2, $^)) --no_divide $@ --t_rows 0 10 31

$(FIGS_FILTER)/with_divide.jpeg: $(FilterPlotscripts)/forecast_update.py $(FILTER_DATA)/ddays/dict.pkl
	mkdir -p $(@D)
	python $< $(dir $(word 2, $^)) --with_divide $@ --t_rows 70 74 78

# For ddays25.  Scale matches hmm figure with many states
$(FIGS_FILTER)/entropy_particle.pdf: $(FilterPlotscripts)/entropy_particle.py $(FILTER_DATA)/particle_1k/dict.pkl
	mkdir -p $(@D)
	python $< --ylim 0 5.9 --dir_template $(FILTER_DATA)/{} --save $@ particle_1k

# For filter.tex.  Has hat h on right
$(FIGS_FILTER)/entropy_filter.pdf: $(FilterPlotscripts)/entropy_particle.py $(FILTER_DATA)/particle_1k/dict.pkl
	mkdir -p $(@D)
	python $< --show_h_hat --dir_template $(FILTER_DATA)/{} --save $@ particle_1k

$(FIGS_FILTER)/clouds10XOK.jpeg: $(FilterPlotscripts)/plot_clouds.py $(FILTER_DATA)/s_augment10X/0.0003
	mkdir -p $(@D)
	python $^ --start 2625 --save $@

$(FIGS_FILTER)/clouds10Xbad.jpeg: $(FilterPlotscripts)/plot_clouds.py $(FILTER_DATA)/s_augment10X/0.0001
	mkdir -p $(@D)
	python $^ --start 2625 --save $@

$(FIGS_FILTER)/particles_a.pdf: $(FilterPlotscripts)/ddays_plot_a.py $(FILTER_DATA)/ddays/dict.pkl
	mkdir -p $(@D)
	python $< $(dir $(word 2, $^)) $@

$(FIGS_FILTER)/particles_b.pdf: $(FilterPlotscripts)/ddays_plot_b.py $(FILTER_DATA)/ddays/dict.pkl
	mkdir -p $(@D)
	python $< $(dir $(word 2, $^)) --start 72 $@

# Local Variables:
# mode: makefile
# End:

# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

FILTER_DATA = $(BUILD)/derived_data/synthetic/filter
FIGS_FILTER = $(BUILD)/figs/filter
FilterPlotscripts = $(ROOT)/src/plotscripts/filter

$(FIGS_FILTER)/%_filter.pdf: $(FILTER_DATA)/%_data $(FilterPlotscripts)/filter_fig.py
	mkdir -p $(@D)
	python $(FilterPlotscripts)/filter_fig.py --sample_ratio 5 $< $@

$(FIGS_FILTER)/%_smooth.pdf: $(FILTER_DATA)/%_data $(FilterPlotscripts)/smooth_fig.py
	mkdir -p $(@D)
	python $(FilterPlotscripts)/smooth_fig.py $< $@

$(FIGS_FILTER)/distribution.pdf: $(FilterPlotscripts)/distribution_fig.py $(FILTER_DATA)/distribution_data
	mkdir -p $(@D)
	python $^ $@

$(FIGS_FILTER)/log_likelihood.pdf: $(FilterPlotscripts)/log_likelihood_fig.py $(FILTER_DATA)/log_likelihood_data
	mkdir -p $(@D)
	python $^ $@

$(FIGS_FILTER)/filter_b.pdf: $(FilterPlotscripts)/forecast_update.py $(BOUNDS_DATA)/ddays
	mkdir -p $(@D)
	python $^ --start 72 $@

$(FIGS_FILTER)/no_divide.pdf: $(FilterPlotscripts)/forecast_update.py $(BOUNDS_DATA)/ddays
	mkdir -p $(@D)
	python $^ --no_divide $@ --t_rows 0 10 31

$(FIGS_FILTER)/with_divide.pdf: $(FilterPlotscripts)/forecast_update.py $(BOUNDS_DATA)/ddays
	mkdir -p $(@D)
	python $^ --with_divide $@ --t_rows 70 74 78

$(FIGS_FILTER)/entropy_filter.pdf: $(FilterPlotscripts)/forecast_update.py $(BOUNDS_DATA)/particle_1k
	mkdir -p $(@D)
	python $^ --entropy $@

# Local Variables:
# mode: makefile
# End:

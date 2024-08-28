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

# Local Variables:
# mode: makefile
# End:

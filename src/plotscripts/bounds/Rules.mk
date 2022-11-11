# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

BOUNDS_DATA = $(BUILD)/derived_data/synthetic/bounds
FIGS_BOUNDS = $(BUILD)/figs/bounds
BoundsPlotscripts = $(ROOT)/src/plotscripts/bounds


$(FIGS_BOUNDS)/ToyTS1.pdf: $(BoundsPlotscripts)/plot.py $(BOUNDS_DATA)/data_h_view
	python $< --ToyTS1 $@ $(BOUNDS_DATA)/data_h_view

$(FIGS_BOUNDS)/ToyStretch.pdf: $(BoundsPlotscripts)/plot.py $(BOUNDS_DATA)/data_h_view
	python $< --ToyStretch $@ $(BOUNDS_DATA)/data_h_view

$(FIGS_BOUNDS)/ToyH.pdf: $(BoundsPlotscripts)/plot_toy_h.py $(BOUNDS_DATA)/toy_h.data
	python $^ $@

$(FIGS_BOUNDS)/benettin.pdf: $(BoundsPlotscripts)/benettin_plot.py $(BOUNDS_DATA)/BenettinData
	python $^ $@

$(FIGS_BOUNDS)/LikeLor.pdf: $(BoundsPlotscripts)/plot_like_lor.py $(BOUNDS_DATA)/like_lor_data
	python $^ $@

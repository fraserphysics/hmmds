# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

BOUNDS_DATA = $(BUILD)/derived_data/synthetic/bounds
FIGS_BOUNDS = $(BUILD)/figs/bounds
BoundsPlotscripts = $(ROOT)/src/plotscripts/bounds

$(FIGS_BOUNDS)/ToyTS1.pdf: $(BoundsPlotscripts)/toy.py $(BOUNDS_DATA)/data_h_view
	mkdir -p $(@D)
	python $< --ToyTS1 $@ $(BOUNDS_DATA)/data_h_view

$(FIGS_BOUNDS)/ToyStretch.pdf: $(BoundsPlotscripts)/toy.py $(BOUNDS_DATA)/data_h_view
	mkdir -p $(@D)
	python $< --t_view 20 --ToyStretch $@ $(BOUNDS_DATA)/data_h_view

$(FIGS_BOUNDS)/ToyH.pdf: $(BoundsPlotscripts)/toy_h.py $(BOUNDS_DATA)/toy_h $(BOUNDS_DATA)/benettin
	mkdir -p $(@D)
	python $^ $@

$(FIGS_BOUNDS)/benettin.pdf: $(BoundsPlotscripts)/benettin.py $(BOUNDS_DATA)/benettin
	mkdir -p $(@D)
	python $^ $@

$(FIGS_BOUNDS)/LikeLor.pdf: $(BoundsPlotscripts)/like_lor.py $(BOUNDS_DATA)/like_lor $(BOUNDS_DATA)/benettin
	mkdir -p $(@D)
	python $^ $@

$(FIGS_BOUNDS)/particles_a.pdf: $(BoundsPlotscripts)/ddays_plot_a.py $(BOUNDS_DATA)/ddays
	mkdir -p $(@D)
	python $^ $@

$(FIGS_BOUNDS)/particles_b.pdf: $(BoundsPlotscripts)/ddays_plot_b.py $(BOUNDS_DATA)/ddays
	mkdir -p $(@D)
	python $^ --start 72 $@

$(FIGS_BOUNDS)/entropy_particle.pdf: $(BoundsPlotscripts)/entropy_particle.py $(BOUNDS_DATA)/particle_1k
	mkdir -p $(@D)
	python $^ $@

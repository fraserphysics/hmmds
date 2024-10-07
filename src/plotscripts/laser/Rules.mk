# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the root of this project and
# BUILD is where derived results go.

LASER_DATA = $(BUILD)/derived_data/synthetic
FIGS_LASER = $(BUILD)/figs/laser
LaserPlotscripts = $(ROOT)/src/plotscripts/laser

$(FIGS_LASER)/%_plot.pdf: $(LaserPlotscripts)/plot.py $(LASER_DATA)/%.plot_data
	mkdir -p $(FIGS_LASER)
	python $^ $@

$(FIGS_LASER)/forecast_errors.pdf: $(LaserPlotscripts)/cumulative.py $(LASER_DATA)/pf_hand_noise.plot_data
	mkdir -p $(FIGS_LASER)
	python $^ $@

$(FIGS_LASER)/LaserLikeOptTS.pdf: $(LaserPlotscripts)/laser_figures.py $(LASER_DATA)/LaserLikeOptTS
	mkdir -p $(FIGS_LASER)
	python $< --LaserLP5 $(word 2, $^) $@

$(FIGS_LASER)/Laser%.pdf: $(LaserPlotscripts)/laser_figures.py $(LASER_DATA)/Laser%
	mkdir -p $(FIGS_LASER)
	python $< --Laser$* $(LASER_DATA)/Laser$* $@


# Local Variables:
# mode: makefile
# End:

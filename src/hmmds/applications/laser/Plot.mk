# Plot.mk

# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the root of this project and
# BUILD is where derived results go.

# To develop, assign these to be local in an interim makefile

# LASER_DATA = $(BUILD)/derived_data/synthetic
# FIGS_LASER = $(BUILD)/figs/laser
# LaserPlotscripts = $(ROOT)/plotscripts/laser

# BookFigs = LaserLP5.pdf LaserLogLike.pdf LaserStates.pdf LaserForecast.pdf LaserHist.pdf

# FIGURES = gui_plot.pdf ekf_powell250_plot.pdf ekf_powell2876_plot.pdf \
# pf_ekf250_plot.pdf pf_opt_noise_plot.pdf pf_hand_noise_plot.pdf forecast_errors.pdf

$(FIGS_LASER)/%_plot.pdf: $(LaserPlotscripts)/plot.py $(LASER_DATA)%.plot_data
	python $^ $@

$(FIGS_LASER)/forecast_errors.pdf: $(LaserPlotscripts)/cumulative.py pf_hand_noise.plot_data
	python $^ $@

$(FIGS_LASER)/Laser%.pdf: $(LaserPlotscripts)/laser_figures.py Laser%
	python $< --Laser$* Laser$* $@


# Local Variables:
# mode: makefile
# End:

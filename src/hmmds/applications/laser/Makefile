SHELL=bash

# f_max=-213.15338306475672
# success=False
# message=Desired error not necessarily achieved due to precision loss.
# iterations=2

#real	138m56.496s
#user	415m19.560s
#sys	0m16.864s

FIGS_LASER = .
LASER_CODE = .
LASER_DATA = .
LaserPlotscripts = .
LP5DAT = LP5.DAT

laser_fit.pdf:

include Rules.mk
include Plot.mk

BookFigs = LaserLP5.pdf LaserLogLike.pdf LaserStates.pdf LaserForecast.pdf LaserHist.pdf

FIGURES = gui_plot.pdf ekf_powell250_plot.pdf ekf_powell2876_plot.pdf \
pf_ekf250_plot.pdf pf_opt_noise_plot.pdf pf_hand_noise_plot.pdf forecast_errors.pdf

laser_fit.pdf: laser_fit.tex $(FIGURES) $(BookFigs)
	pdflatex laser_fit.tex
	pdflatex laser_fit.tex

# Rules.mk: Needs BUILD and TEX

LASER_TEX_OUT = $(BUILD)/TeX/laser
LASER_TEX_IN = $(TEX)/laser
LASER_FIG_DIR = $(BUILD)/figs/laser/

FIGS = $(addsuffix .pdf, $(addprefix $(LASER_FIG_DIR), \
LaserLP5 LaserLogLike LaserStates LaserForecast LaserHist gui_plot \
ekf_powell250_plot ekf_powell2876_plot pf_ekf250_plot pf_opt_noise_plot \
pf_hand_noise_plot forecast_errors))

$(LASER_TEX_OUT)/laser_fit.pdf: $(LASER_TEX_IN)/laser_fit.tex $(FIGS)
	mkdir -p $(LASER_TEX_OUT)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
pdflatex --output-directory $(LASER_TEX_OUT) $(LASER_TEX_IN)/laser_fit.tex; \
pdflatex --output-directory $(LASER_TEX_OUT) $(LASER_TEX_IN)/laser_fit.tex;

# Local Variables:
# mode: makefile
# End:

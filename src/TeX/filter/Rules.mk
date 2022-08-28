# Rules.mk: Needs BUILD and TEX

FILTER_TEX_OUT = $(BUILD)/TeX/filter
FILTER_TEX_IN = $(TEX)/filter
FILTER_FIG_DIR = $(BUILD)/figs/filter/

FIGS = $(addsuffix .pdf, $(addprefix $(FILTER_FIG_DIR), distribution	\
linear_sde_filter lorenz_particle_filter linear_map_filter		\
linear_sde_smooth lorenz_smooth linear_map_smooth log_likelihood	\
linear_particle_filter lorenz_filter ))

$(FILTER_TEX_OUT)/filter.pdf: $(FILTER_TEX_IN)/filter.tex $(FIGS)
	mkdir -p $(FILTER_TEX_OUT)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
pdflatex --output-directory $(FILTER_TEX_OUT) $(FILTER_TEX_IN)/filter.tex; \
pdflatex --output-directory $(FILTER_TEX_OUT) $(FILTER_TEX_IN)/filter.tex;

# Local Variables:
# mode: makefile
# End:

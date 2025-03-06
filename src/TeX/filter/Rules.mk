# Rules.mk: Needs BUILD and TEX

FILTER_TEX_OUT = $(BUILD)/TeX/filter
FILTER_TEX_IN = $(TEX)/filter
FILTER_FIG_DIR = $(BUILD)/figs/filter/

FIGS = $(addsuffix .pdf, $(addprefix $(FILTER_FIG_DIR), distribution	\
linear_sde_filter lorenz_particle_filter linear_map_filter		\
linear_sde_smooth lorenz_smooth linear_map_smooth log_likelihood	\
linear_particle_filter lorenz_filter filter_b entropy_filter))

$(FILTER_TEX_OUT)/filter.pdf: $(FILTER_TEX_IN)/filter.tex $(FIGS)
	mkdir -p $(FILTER_TEX_OUT)
	export TEXINPUTS=$(FILTER_TEX_IN)//:$(abspath $(BUILD))//:; \
latexmk --outdir=$(@D) -pdflatex filter.tex

# Local Variables:
# mode: makefile
# End:

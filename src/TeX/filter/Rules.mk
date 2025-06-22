# Rules.mk: Needs BUILD and TEX

FILTER_TEX_OUT = $(BUILD)/TeX/filter
FILTER_TEX_IN = $(TEX)/filter
FILTER_FIG_DIR = $(BUILD)/figs/filter/

FILTER_FIGS_pdf = $(addsuffix .pdf, $(addprefix $(FILTER_FIG_DIR), distribution  \
linear_sde_filter lorenz_particle_filter linear_map_filter		  \
linear_sde_smooth lorenz_smooth linear_map_smooth log_likelihood	  \
linear_particle_filter lorenz_filter entropy_filter))

FILTER_FIGS_jpeg = $(addsuffix .jpeg, $(addprefix $(FILTER_FIG_DIR), with_divide \
no_divide clouds10XOK clouds10Xbad))

FILTER_TABLES = $(addsuffix .tex, $(addprefix $(FILTER_DATA)/, s_augment s_augment10X h_max edge_max margin r_extra r_threshold))

$(FILTER_TEX_OUT)/filter.pdf: $(FILTER_TEX_IN)/filter.tex $(FILTER_TABLES) $(FILTER_FIGS_jpeg) $(FILTER_FIGS_pdf) 
	mkdir -p $(FILTER_TEX_OUT)
	export TEXINPUTS=$(FILTER_TEX_IN)//:$(abspath $(BUILD))//:; \
latexmk --outdir=$(@D) -pdflatex filter.tex
	touch $@

# Local Variables:
# mode: makefile
# End:

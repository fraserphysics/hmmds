# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

#DERIVED_APNEA_DATA = $(BUILD)/derived_data/apnea
#APNEA_FIG_DIR = $(BUILD)/figs/apnea
#APNEA_PLOTSCRIPTS = $(ROOT)/src/plotscripts/apnea

Apnea_Build = $(BUILD)/TeX/apnea
ApneaTeX = $(ROOT)/src/TeX/apnea
# This file is in the ApneaTeX directory

# Data built elsewhere
#RAW_DATA = /mnt/precious/home/andy_nix/projects/dshmm/raw_data
#EXPERT =  $(RAW_DATA)/apnea/summary_of_training
#PHYSIONET_WFDB = $(ROOT)/raw_data/apnea/apnea-ecg-database

#MODELS = $(BUILD)/derived_data/apnea/models
#ECG = $(MODELS)/ECG

########################Build hand_opt.pdf####################################

LIST_ERRORS = ar fs lpp rc rw rs
ERRORS = $(addsuffix .pdf, $(addprefix $(APNEA_FIG_DIR)/errors_vs_, $(LIST_ERRORS)))

APNEA_TEX_INCLUDES = $(addsuffix .tex, $(addprefix $(DERIVED_APNEA_DATA)/, score test_score))

HANDOPT_FIGS = $(ERRORS) \
$(addsuffix .pdf, $(addprefix $(APNEA_FIG_DIR)/, threshold viz))
$(Apnea_Build)/hand_opt.pdf: $(ApneaTeX)/hand_opt.tex $(HANDOPT_FIGS) $(APNEA_TEX_INCLUDES)
	mkdir -p  $(@D)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
	pdflatex --output-directory=$(@D) $< ; pdflatex --output-directory=$(@D) $<

# Local Variables:
# mode: makefile
# End:

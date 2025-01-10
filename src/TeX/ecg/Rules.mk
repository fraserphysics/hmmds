# Rules.mk: This file can be included by a makefile anywhere as long
# as the following are defined:

# ROOT         : The root of this project
# BUILD        : Directory tree for derived results
# PICKLED_ECG  : Directory of ecg files
# ECG_DERIVED  : Holds, eg, self_table.tex
# ECG_FIG_DIR  : defined in plotscripts/ecg/Rules.mk

ECG_TeX = $(ROOT)/src/TeX/ecg
# This file is in the ECG_TeX directory

BUILD_ECG_TeX = $(BUILD)/TeX/ecg

ECG_FIGLIST = $(addsuffix .pdf, $(addprefix $(ECG_FIG_DIR)/, \
a01_trained_AR3_states_70 a01_trained_AR3_states_71 a01a19c02 train_log \
like_a14_x07 simulated))

$(ECG_TeX)/ecg.pdf: ecg.tex $(ECG_FIGLIST) $(ECG_DERIVED)/table.tex $(ECG_DERIVED)/self_table.tex
	mkdir -p $(@D)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
        pdflatex --output-directory=$(@D) $< ; pdflatex --output-directory=$(@D) $<

################################################################################

DS23FIGS = $(addsuffix .pdf, $(addprefix $(ECG_FIG_DIR)/, elgendi constant_a03 a03_states_56 a01c02_states simulated ecg_hmm)) $(ECG_FIG_DIR)/ecg_hmm.pdf_t

$(BUILD_ECG_TeX)/ds23.pdf: $(ECG_TeX)/ds23.tex $(DS23FIGS)
	mkdir -p $(@D)
	export TEXINPUTS=$(abspath $(ROOT))//:; pdflatex --output-directory=$(@D) $<; pdflatex --output-directory=$(@D) $<

# Local Variables:
# mode: makefile
# End:

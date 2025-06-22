# Rules.mk: This file can be included by a makefile anywhere as long
# as the following are defined:

# ROOT         : The root of this project
# BUILD        : Directory tree for derived results
# PICKLED_ECG  : Directory of ecg files
# ECG_DERIVED  : Holds, eg, self_table.tex
# ECG_FIG_DIR  : defined in plotscripts/ecg/Rules.mk

TeX_DS25 = $(ROOT)/src/TeX/siamDS25
# This file is in the TeX_DS25 directory

BUILD_DS25 = $(BUILD)/TeX/ds25

DS25_FIGLIST = $(DS25_XFIGS) $(DS25_APNEA_FIGS) $(DS25_ECG_FIGS)

$(BUILD_DS25)/ds25.pdf: $(TeX_DS25)/ds25.tex $(DS25_FIGLIST)
	mkdir -p $(@D)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
 pdflatex --output-directory=$(@D) $< ; pdflatex --output-directory=$(@D) $<

################################################################################

# Local Variables:
# mode: makefile
# End:

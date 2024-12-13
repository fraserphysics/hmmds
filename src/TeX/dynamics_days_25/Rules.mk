# Rules.mk: Derived from TeX/book/Rules.mk.  The poster target is
# $(BUILD)/TeX/dynamics_days_25/poster.pdf.  Figures for the poster
# can be anywhere in the TEXINPUTS defined in the rule for the poster.
# This file can be included by a makefile anywhere as long as ROOT and
# BUILD are defined.  ROOT is the top directory of dshmm, DD25 is the
# directory where this file is located and BUILD is where results are
# written.

# This directory
DD25SRC = $(ROOT)/src/TeX/dynamics_days_25

DD25_FIGS = $(INTRODUCTION_FIGS) \
$(LASER_FIGS) \
$(INTRODUCTION_XFIGS) \
$(BOUNDS_FIGS) \
$(BOUNDS_XFIGS)

$(BUILD)/TeX/dynamics_days_25/poster.pdf: $(DD25SRC)/poster.tex $(DD25PosterFigs)
	mkdir -p $(@D)
	export TEXINPUTS=$(DD25SRC)//:$(abspath $(BUILD))//:; \
latexmk --outdir=$(@D) -pdflatex poster.tex;

# Local Variables:
# mode: makefile
# End:

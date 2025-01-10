# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the top directory of
# dshmm, TEX_SKELTON is the directory where this file is located and
# BUILD is where results are written.

SKELETON_OUT = $(BUILD)/TeX/skeleton
TEX_SKELETON = $(ROOT)/src/TeX/skeleton

SKELETON_FIGS = $(INTRODUCTION_FIGS) \
$(LASER_FIGS) \
$(INTRODUCTION_XFIGS) \
$(BASIC_ALGORITHMS_FIGS) \
$(BASIC_ALGORITHMS_XFIGS) \
$(VARIANTS_FIGS) \
$(VARIANTS_XFIGS) \
$(BOUNDS_FIGS) \
$(BOUNDS_XFIGS) \
$(APNEA_FIGS) \
$(APNEA_XFIGS) \
$(ECG_FIGS) \
$(ECG_XFIGS)

$(SKELETON_OUT)/figures.pdf: $(TEX_SKELETON)/figures.tex $(TEX_BOOK)/apnea.tex $(SKELETON_FIGS) $(TEX_INCULDES)
	mkdir -p $(@D)
	export TEXINPUTS=$(TEX_SKELETON):$(abspath $(BUILD))//:;  \
pdflatex --output-directory $(@D) $(TEX_SKELETON)/figures.tex; \
pdflatex --output-directory $(@D) $(TEX_SKELETON)/figures.tex;

# Local Variables:
# mode: makefile
# End:

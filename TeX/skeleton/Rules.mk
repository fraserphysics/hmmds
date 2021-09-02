# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, TEX_SKELTON and BUILD are defined.  ROOT is the top directory of
# dshmm, TEX_SKELTON is the directory where this file is located and
# BUILD is where results are written.

TEX_OUT = $(BUILD)/TeX/skeleton

# Define a function with arguments prefix=$(1) names=$(2) that
# prepends prefix and appends pdf and pdf_t to names
ADD_PDF_PDF_T = $(addprefix $(1)/, $(addsuffix .pdf, $(2)) $(addsuffix .pdf_t, $(2)))

INTRODUCTION_FIGS = $(addprefix $(BUILD)/figs/introduction/, TSintro.pdf)
INTRODUCTION_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/introduction, Markov_mm Markov_dhmm Markov_dhmm_net)

BASIC_ALGORITHMS_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/basic_algorithms, forward viterbiB)

SKELETON_FIGS = $(INTRODUCTION_FIGS) $(INTRODUCTION_XFIGS) $(BASIC_ALGORITHMS_XFIGS)

TEX_SKELETON = $(ROOT)/TeX/skeleton

$(TEX_OUT)/figures.pdf: $(TEX_SKELETON)/figures.tex $(SKELETON_FIGS)
	mkdir -p $(TEX_OUT)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
pdflatex --output-directory $(TEX_OUT) $(TEX_SKELETON)/figures.tex; \
pdflatex --output-directory $(TEX_OUT) $(TEX_SKELETON)/figures.tex;

# Local Variables:
# mode: makefile
# End:

# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, TEX_SKELTON and BUILD are defined.  ROOT is the top directory of
# dshmm, TEX_SKELTON is the directory where this file is located and
# BUILD is where results are written.

TEX_OUT = $(BUILD)/TeX/skeleton

# Define a function with arguments prefix=$(1) names=$(2) that
# prepends prefix and appends pdf and pdf_t to names
ADD_PDF_PDF_T = $(addprefix $(1)/, $(addsuffix .pdf, $(2)) $(addsuffix .pdf_t, $(2)))

INTRODUCTION_FIGS = $(addsuffix .pdf, $(addprefix $(BUILD)/figs/introduction/, \
TSintro STSintro Statesintro))
INTRODUCTION_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/introduction, Markov_mm Markov_dhmm Markov_dhmm_net nonmm)

BASIC_ALGORITHMS_FIGS = $(addsuffix .pdf, $(addprefix $(BUILD)/figs/basic_algorithms/, \
TrainChar GaussMix EM))
BASIC_ALGORITHMS_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/basic_algorithms, forward viterbiB sequenceMAP)

VARIANTS_FIGS =  $(addprefix $(BUILD)/figs/variants/, SGO_bcd $(addsuffix .pdf, VARGstates))
VARIANTS_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/variants, ScalarGaussian)

BOUNDS_FIGS =  $(addprefix $(BUILD)/figs/bounds/,  $(addsuffix .pdf, \
ToyTS1 ToyStretch ToyH benettin LikeLor))
BOUNDS_XFIGS =  $(call ADD_PDF_PDF_T, $(BUILD)/figs/bounds, QR)

APNEA_XFIGS =  $(call ADD_PDF_PDF_T, $(BUILD)/figs/apnea, class_net)

SKELETON_FIGS = $(INTRODUCTION_FIGS) \
$(INTRODUCTION_XFIGS) \
$(BASIC_ALGORITHMS_FIGS) \
$(BASIC_ALGORITHMS_XFIGS) \
$(VARIANTS_FIGS) \
$(VARIANTS_XFIGS) \
$(BOUNDS_FIGS) \
$(BOUNDS_XFIGS) \
$(APNEA_XFIGS)

TEX_SKELETON = $(TEX)/skeleton

$(TEX_OUT)/figures.pdf: $(TEX_SKELETON)/figures.tex $(SKELETON_FIGS)
	mkdir -p $(TEX_OUT)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
pdflatex --output-directory $(TEX_OUT) $(TEX_SKELETON)/figures.tex; \
pdflatex --output-directory $(TEX_OUT) $(TEX_SKELETON)/figures.tex;

# Local Variables:
# mode: makefile
# End:

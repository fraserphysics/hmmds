# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the top directory of
# dshmm, TEX_BOOK is the directory where this file is located and
# BUILD is where results are written.

BOOK_OUT = $(BUILD)/TeX/book

# Define a function with arguments prefix=$(1) names=$(2) that
# prepends prefix and appends pdf and pdf_t to names
ADD_PDF_PDF_T = $(addprefix $(1)/, $(addsuffix .pdf, $(2)) $(addsuffix .pdf_t, $(2)))

LASER_FIG_DIR = $(BUILD)/figs/laser/
LASER_FIGS = $(addsuffix .pdf, $(addprefix $(LASER_FIG_DIR), \
LaserLP5 LaserLogLike LaserStates LaserForecast LaserHist))

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

APNEA_FIGS =  $(addprefix $(BUILD)/figs/apnea/,  $(addsuffix .pdf, \
a03erA a03erN a03HR sgram explore viz errors_vs_fs explore))
APNEA_XFIGS =  $(call ADD_PDF_PDF_T, $(BUILD)/figs/apnea, class_net)

ECG_FIGS =  $(addprefix $(BUILD)/figs/ecg/,  $(addsuffix .pdf, \
elgendi a03a10b03c02 constant_a03 a01c02_states simulated ecg2hr))
ECG_XFIGS =  $(call ADD_PDF_PDF_T, $(BUILD)/figs/ecg/, ecg_hmm)

TEX_INCLUDES = $(addsuffix .tex, $(addprefix $(BUILD)/derived_data/apnea/, score test_score))
TEX_BOOK = $(ROOT)/src/TeX/book

BOOK_FIGS = $(INTRODUCTION_FIGS) \
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

BOOK_CHAPTERS = $(addprefix $(TEX_BOOK)/, $(addsuffix .tex, \
algorithms apnea appendix continuous introduction main toys variants))

VALUE_FILES = $(addprefix $(BOOK_OUT)/, $(addsuffix .tex, \
apnea_values decoded_menken toy_values synthetic_values text_values))

BOOK_SENTINELS = $(SYNTHETIC_DATA)/SGO

$(BOOK_OUT)/main.pdf: $(BOOK_CHAPTERS) $(BOOK_FIGS) $(TEX_INCLUDES) $(VALUE_FILES) $(BOOK_SENTINELS)
	mkdir -p $(@D)
	export TEXINPUTS=$(TEX_BOOK)//:$(abspath $(BUILD))//:; \
export BIBINPUTS=$(TEX_BOOK)//:; export BSTINPUTS=$(TEX_BOOK)//:; \
latexmk --outdir=$(@D) -pdflatex main.tex;

# Note that latexmk seems to detect changes in dependencies without
# using file change times.  Thus deleting and regenerating a figure
# will not cause the document to be rebuilt if the regenerated figure
# is the same as the old one.


# Local Variables:
# mode: makefile
# End:

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
TSintro STSintro Statesintro GraphStates))
INTRODUCTION_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/introduction, Markov_mm Markov_dhmm Markov_dhmm_net nonmm)

BASIC_ALGORITHMS_FIGS = $(addsuffix .pdf, $(addprefix $(BUILD)/figs/basic_algorithms/, \
TrainChar EM))
BASIC_ALGORITHMS_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/basic_algorithms, forward viterbiB sequenceMAP)

VARIANTS_FIGS =  $(addprefix $(BUILD)/figs/variants/, SGO_bcd $(addsuffix .pdf, VARGstates))
VARIANTS_XFIGS = $(call ADD_PDF_PDF_T, $(BUILD)/figs/variants, ScalarGaussian EMxfig)

BOUNDS_FIGS =  $(addprefix $(BUILD)/figs/bounds/,  $(addsuffix .pdf, \
ToyTS1 ToyStretch ToyH benettin LikeLor))
BOUNDS_XFIGS =  $(call ADD_PDF_PDF_T, $(BUILD)/figs/bounds, QR)

APNEA_FIGS =  $(addprefix $(BUILD)/figs/apnea/,  $(addsuffix .pdf, \
a03erA a03erN a03HR explore viz errors_vs_fs explore) sgram.jpg)
APNEA_XFIGS =  $(call ADD_PDF_PDF_T, $(BUILD)/figs/apnea, class_net)

ECG_FIGS =  $(addprefix $(BUILD)/figs/ecg/,  $(addsuffix .pdf, \
elgendi a03a10b03c02 constant_a03 a01c02_states simulated ecg2hr))
ECG_XFIGS =  $(call ADD_PDF_PDF_T, $(BUILD)/figs/ecg/, ecg_hmm)

TEX_INCLUDES = $(addsuffix .tex, $(addprefix $(BUILD)/derived_data/apnea/, score test_score))
# Using absolute paths has resolved trouble with the TeX programs
TEX_BOOK = $(abspath $(ROOT)/src/TeX/book)

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
algorithms apnea particles appendix continuous introduction main toys variants))

VALUE_FILES = $(addprefix $(BOOK_OUT)/, $(addsuffix .tex, \
apnea_values decoded_menken toy_values synthetic_values text_values))

BOOK_SENTINELS = $(SYNTHETIC_DATA)/SGO

# The book should have the following parts in this order: Chapters;
# Appendices; Notation/nomenclature; Bibliography; Index.
# The following produced those sections on 2025-02-20.

$(BOOK_OUT)/main.pdf: $(TEX_BOOK)/main.tex $(BOOK_CHAPTERS) $(BOOK_FIGS) $(TEX_INCLUDES) $(VALUE_FILES) $(BOOK_SENTINELS)
	mkdir -p $(@D)
	export TEXINPUTS=$(TEX_BOOK)//:$(abspath $(BUILD))//:; \
export BIBINPUTS=$(TEX_BOOK)//:; export BSTINPUTS=$(TEX_BOOK)//:; \
cd $(@D) ; \
pdflatex --output-directory=$(abspath $(@D)) $< ; \
# The next line builds main.nls.  I needed to call makeindex from the \
# build dir \
makeindex main.nlo -s nomencl.ist -o main.nls; \
# The next line makes main.idx \
makeindex main.idx ; \
bibtex main.aux ; \
pdflatex --output-directory=$(abspath $(@D)) $< ; \
# The next pdflatex gets "Notation" into the table of contents \
pdflatex --output-directory=$(abspath $(@D)) $<

# Note that latexmk seems to detect changes in dependencies without
# using file change times.  Thus deleting and regenerating a figure
# will not cause the document to be rebuilt if the regenerated figure
# is the same as the old one.  I've set up emacs to call latexmk from
# auctex.  So that works as long as the index, notation and bib files
# have already been built using make and this file.

# I replaced the line below with the pdflatex stuff above because I was having trouble with nomencl
#latexmk --outdir=$(@D) -pdflatex main.tex;


# Local Variables:
# mode: makefile
# End:

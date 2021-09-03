# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

INTRODUCTION = $(BUILD)/figs/introduction

BASIC_ALGORITHMS = $(BUILD)/figs/basic_algorithms

VARIANTS = $(BUILD)/figs/variants

BOUNDS = $(BUILD)/figs/bounds

APNEA = $(BUILD)/figs/apnea

XFIGS = $(ROOT)/plotscripts/xfigs
# XFIGS is _this_ directory

# Define a function with arguments prefix=$(1) names=$(2) that
# prepends prefix and appends pdf and pdf_t to names
ADD_PDF_PDF_T = $(addprefix $(1)/, $(addsuffix .pdf, $(2)) $(addsuffix .pdf_t, $(2)))

# The following *XFIGS variables list all of the targets for which this
# makefile is responsible.

INTRODUCTION_XFIGS = $(call ADD_PDF_PDF_T, $(INTRODUCTION), Markov_mm Markov_dhmm Markov_dhmm_net nonmm)

BASIC_ALGORITHMS_XFIGS = $(call ADD_PDF_PDF_T, $(BASIC_ALGORITHMS), forward viterbiB sequenceMAP)

VARIANTS_XFIGS = $(call ADD_PDF_PDF_T, $(VARIANTS), ScalarGaussian)

BOUNDS_XFIGS = $(call ADD_PDF_PDF_T, $(BOUNDS), QR)

APNEA_XFIGS =  $(call ADD_PDF_PDF_T, $(APNEA), class_net)

# The function double_rule defines pattern rules translate an xfig
# file into files suitable to include in a LaTeX file.  The mysterious
# agruments to fig2dev are: -p name of the pdf file to be overlaid, -F
# Don't set the font face, -series, or style.  In the rules for
# %.pdf_t, I use the absolute path for the -p argument so that LaTeX
# can find the $*.pdf file from any context.

define double_rule
# $(1), the first argument to this function, is the traget directory
$(1)/%.pdf : $(XFIGS)/%.fig
	mkdir -p $(1)
	fig2dev -L pdftex -F $$< $$@
$(1)/%.pdf_t: $(XFIGS)/%.fig
	mkdir -p $(1)
	fig2dev -L pdftex_t -p $(abspath $(1)/$$*.pdf) $$< $$@
endef

$(eval $(call double_rule, $(INTRODUCTION)))
$(eval $(call double_rule, $(BASIC_ALGORITHMS)))
$(eval $(call double_rule, $(VARIANTS)))
$(eval $(call double_rule, $(BOUNDS)))
$(eval $(call double_rule, $(APNEA)))

# The remaining rules are for figures for which the pattern rules are
# not adequate.

# In Markov_mm.fig, the basic Markov graph is on layer 50 and the
# output features are on layer 40.

# The new arguments to fig2dev are: -K Bounding box limited to layers
# used; -D specify layers

$(INTRODUCTION)/Markov_mm.pdf_t: $(XFIGS)/Markov_mm.fig
	fig2dev -L pdftex_t -D+50 -K -p $(abspath $(INTRODUCTION)/Markov_mm.pdf) $< $@
$(INTRODUCTION)/Markov_mm.pdf: $(XFIGS)/Markov_mm.fig
	fig2dev -L pdftex -D+50 -K $< $@

$(INTRODUCTION)/Markov_dhmm.pdf_t: $(XFIGS)/Markov_mm.fig
	fig2dev -L pdftex_t -D+40,50 -p $(abspath $(INTRODUCTION)/Markov_dhmm.pdf) $< $@
$(INTRODUCTION)/Markov_dhmm.pdf: $(XFIGS)/Markov_mm.fig
	fig2dev -L pdftex -D+40,50 $< $@

# Local Variables:
# mode: makefile
# End:

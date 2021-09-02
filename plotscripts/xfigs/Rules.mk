# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

INTRODUCTION = $(BUILD)/figs/introduction

BASIC_ALGORITHMS = $(BUILD)/figs/basic_algorithms

XFIGS = $(ROOT)/plotscripts/xfigs
# XFIGS is _this_ directory

# Define a function with arguments prefix=$(1) names=$(2) that
# prepends prefix and appends pdf and pdf_t to names
ADD_PDF_PDF_T = $(addprefix $(1)/, $(addsuffix .pdf, $(2)) $(addsuffix .pdf_t, $(2)))

# The following *XFIGS variables list all of the targets for which this
# makefile is responsible.

INTRODUCTION_XFIGS = $(call ADD_PDF_PDF_T, $(INTRODUCTION), Markov_mm Markov_dhmm Markov_dhmm_net)

BASIC_ALGORITHMS_XFIGS = $(call ADD_PDF_PDF_T, $(BASIC_ALGORITHMS), forward viterbiB)

# The following pattern rules translate an xfig file into files
# suitable to include in a LaTeX file.  The mysterious agruments to
# fig2dev are: -p name of the pdf file to be overlaid, -F Don't set
# the font face, -series, or style.  In the rules for %.pdf_t, I use
# the absolute path for the -p argument so that LaTeX can find the
# $*.pdf file from any context.

$(INTRODUCTION)/%.pdf: $(XFIGS)/%.fig
	fig2dev -L pdftex -F $< $@
$(INTRODUCTION)/%.pdf_t: $(XFIGS)/%.fig
	fig2dev -L pdftex_t -p $(abspath $(INTRODUCTION)/$*.pdf) $< $@

$(BASIC_ALGORITHMS)/%.pdf: $(XFIGS)/%.fig
	fig2dev -L pdftex -F $< $@
$(BASIC_ALGORITHMS)/%.pdf_t: $(XFIGS)/%.fig
	fig2dev -L pdftex_t -p $(abspath $(BASIC_ALGORITHMS)/$*.pdf) $< $@

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

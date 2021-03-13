INTRODUCTION = ../../figs/introduction

BASIC_ALGORITHMS = ../../figs/basic_algorithms

# Define a function with arguments prefix=$(1) names=$(2) that
# prepends prefix and appends pdf and pdf_t to names
ADD_PDF_PDF_T = $(addprefix $(1)/, $(addsuffix .pdf, $(2)) $(addsuffix .pdf_t, $(2)))

INTRODUCTION_FIGS = $(call ADD_PDF_PDF_T, $(INTRODUCTION), Markov_mm Markov_dhmm Markov_dhmm_net)

BASIC_ALGORITHMS_FIGS = $(call ADD_PDF_PDF_T, $(BASIC_ALGORITHMS), forward viterbiB)

.PHONY : test
test: $(INTRODUCTION_FIGS) $(BASIC_ALGORITHMS_FIGS)

$(INTRODUCTION)/%: %
	cp $< $@

$(BASIC_ALGORITHMS)/%: %
	cp $< $@

# The following 2 pattern rules translate an xfig file into files
# suitable to include in a LaTeX file.  The mysterious agruments to
# fig2dev are: -p name of the pdf file to be overlaid, -F Don't set
# the font face, -series, and style
%.pdf: %.fig
	fig2dev -L pdftex -F $< $@
%.pdf_t: %.fig
	fig2dev -L pdftex_t -p $*.pdf $< $@

# The remaining rules are for figures for which the pattern rules are
# not adequate.

# In Markov_mm.fig, the basic Markov graph is on layer 50 and the
# output features are on layer 40.

# The new arguments to fig2dev are: -K Bounding box limited to layers
# used; -D specify layers

Markov_mm.pdf_t: Markov_mm.fig
	fig2dev -L pdftex_t -D+50 -K -p Markov_mm.pdf $< $@
Markov_mm.pdf: Markov_mm.fig
	fig2dev -L pdftex -D+50 -K $< $@

Markov_dhmm.pdf_t: Markov_mm.fig
	fig2dev -L pdftex_t -D+40,50 -p Markov_dhmm.pdf $< $@
Markov_dhmm.pdf: Markov_mm.fig
	fig2dev -L pdftex -D+40,50 $< $@

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo BASIC_ALGORITHMS_FIGS: $(BASIC_ALGORITHMS_FIGS)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

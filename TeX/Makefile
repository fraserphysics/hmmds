# See https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project and http://mirrors.ibiblio.org/CTAN/support/latex-make/latex-make.pdf

FIGS = forward viterbiB Markov_mm Markov_dhmm Markov_dhmm_net
FIG_DEP = $(addsuffix .pdf, $(FIGS)) $(addsuffix .pdf_t, $(FIGS))

bundles.pdf: bundles.tex $(FIG_DEP)
	pdflatex bundles
	pdflatex bundles

# The mysterious agruments to fig2dev are: -p name of the pdf file to
# be overlaid, -F Don't set the font face, -series, and style
%.pdf: %.fig
	fig2dev -L pdftex -F $< $@
%.pdf_t: %.fig
	fig2dev -L pdftex_t -p $*.pdf $< $@

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

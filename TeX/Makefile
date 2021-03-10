# See https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project and http://mirrors.ibiblio.org/CTAN/support/latex-make/latex-make.pdf

bundles.pdf: bundles.tex forward.pdf forward.pdf_t viterbiB.pdf viterbiB.pdf_t
	pdflatex bundles
	pdflatex bundles

%.pdf: %.fig
	fig2dev -L pdftex -n $< -F $< $@
%.pdf_t: %.fig
	fig2dev -L pdftex_t -p $*.pdf $< $@

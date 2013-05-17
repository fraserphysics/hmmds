WEBHOST = amfsr@page.dreamhost.com
WEBDIR = fraserphysics.com/andy/hmmdsbook
WEBPORT = 22

all.www: index.www OT107.www software.www doc.www
	touch all.www

software.www: ../TeX/software.pdf
	scp -P $(WEBPORT) ../TeX/software.pdf $(WEBHOST):$(WEBDIR)/software.pdf
	ssh -p $(WEBPORT) $(WEBHOST) chmod 644 $(WEBDIR)/software.pdf
	touch $@
doc.www: ../doc/build/html
	scp -rP $(WEBPORT) ../doc/build/html $(WEBHOST):$(WEBDIR)/doc
	ssh -p $(WEBPORT) $(WEBHOST) chmod 655 $(WEBDIR)/doc
	touch $@
%.www: %.jpg
	scp -P $(WEBPORT) $*.jpg $(WEBHOST):$(WEBDIR)/$*.jpg
	ssh -p $(WEBPORT) $(WEBHOST) chmod 644 $(WEBDIR)/$*.jpg
	touch $@

%.www: %.pdf
	scp -P $(WEBPORT) $*.pdf $(WEBHOST):$(WEBDIR)/$*.pdf
	ssh -p $(WEBPORT) $(WEBHOST) chmod 644 $(WEBDIR)/$*.pdf
	touch $@

%.www: %.html
	scp -P $(WEBPORT) $*.html $(WEBHOST):$(WEBDIR)/$*.html
	ssh -p $(WEBPORT) $(WEBHOST) chmod 644 $(WEBDIR)/$*.html
	touch $@


clean:
	rm *pdf *ps *dvi *log *aux *.www
#--------------------------------
# Local Variables:
# mode: makefile
# End:

SHELL=bash
# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

TEXT_CODE = $(HMMDS)/applications/text

$(BUILD)/TeX/book/decoded_menken.tex: $(TEXT_CODE)/po_speech.py $(ROOT)/raw_data/menken.txt
	mkdir -p $(@D)
	python $^ $@

# Local Variables:
# mode: makefile
# End:

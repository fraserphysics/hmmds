SHELL=bash
# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

TEXT_CODE = $(HMMDS)/applications/text

# Also makes $(BUILD)/TeX/book/text_values.tex
$(BUILD)/TeX/book/decoded_menken.tex: $(TEXT_CODE)/po_speech.py $(ROOT)/raw_data/menken.txt
	mkdir -p $(@D)
	python $^ --random_seed 10 --n_iterations 100 $(BUILD)/TeX/book/text_values.tex $@

# Hack to circumvent gnu-make inability to make two targets with one rule
$(BUILD)/TeX/book/text_values.tex: $(BUILD)/TeX/book/decoded_menken.tex
	touch $@

# Local Variables:
# mode: makefile
# End:

# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
TrainCharData = $(SYNTHETIC_DATA)/TrainChar
BasicPlotScripts = $(ROOT)/plotscripts/basic_algorithms
FIGS_Basic = $(BUILD)/figs/basic_algorithms

$(FIGS_Basic)/TrainChar.pdf: $(BasicPlotScripts)/TrainChar.py $(TrainCharData)
	mkdir -p $(FIGS_Basic)
	python $< $(TrainCharData) $@

# Local Variables:
# mode: makefile
# End:

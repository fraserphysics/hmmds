# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
TrainCharData = $(SYNTHETIC_DATA)/TrainChar
EMData =  $(SYNTHETIC_DATA)/em.pickle
BasicPlotScripts = $(ROOT)/plotscripts/basic_algorithms
FIGS_Basic = $(BUILD)/figs/basic_algorithms

$(FIGS_Basic)/TrainChar.pdf: $(BasicPlotScripts)/TrainChar.py $(TrainCharData)
	mkdir -p $(FIGS_Basic)
	python $< $(TrainCharData) $@

$(FIGS_Basic)/GaussMix.pdf: $(BasicPlotScripts)/GaussMix.py $(EMData)
	mkdir -p $(FIGS_Basic)
	python $^ $@

# Local Variables:
# mode: makefile
# End:

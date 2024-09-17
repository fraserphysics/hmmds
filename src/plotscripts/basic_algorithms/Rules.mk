# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
TrainCharData = $(SYNTHETIC_DATA)/TrainChar
GaussMixData =  $(SYNTHETIC_DATA)/gauss_mix.pkl
BasicPlotScripts = $(PLOTSCRIPTS)/basic_algorithms
FIGS_Basic = $(BUILD)/figs/basic_algorithms

$(FIGS_Basic)/TrainChar.pdf: $(BasicPlotScripts)/train_char.py $(TrainCharData)
	mkdir -p $(FIGS_Basic)
	python $< $(TrainCharData) $@

$(FIGS_Basic)/GaussMix.pdf: $(BasicPlotScripts)/gauss_mix.py $(GaussMixData)
	mkdir -p $(FIGS_Basic)
	python $^ $@

$(FIGS_Basic)/EM.pdf: $(BasicPlotScripts)/em.py
	mkdir -p $(FIGS_Basic)
	python $^ $@

# Local Variables:
# mode: makefile
# End:

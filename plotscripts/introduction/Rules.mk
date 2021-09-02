# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
TSINTRO = $(SYNTHETIC_DATA)/TSintro
IntroPlotScripts = $(ROOT)/plotscripts/introduction
FIGS_INTRO = $(BUILD)/figs/introduction

$(FIGS_INTRO)/TSintro.pdf: $(IntroPlotScripts)/TSintro.py $(SYNTHETIC_DATA)/lorenz.flag
	mkdir -p $(TSINTRO)
	python $< $(TSINTRO)/fine $(TSINTRO)/coarse $(TSINTRO)/quantized $@

## ALL           : All of the targets that Rules.mk is responsible for
ALL = $(FIGS_INTRO)/TSintro.pdf

# Local Variables:
# mode: makefile
# End:

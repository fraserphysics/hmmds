# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, FIGS_INTRO and IntroPlotScripts are defined.  Scripts will
# read data from locations relative to ROOT, write figures to the
# directory FIGS_INTRO and IntroPlotScripts is the directory where
# this file is located.

TSINTRO = $(ROOT)/derived_data/synthetic/TSintro

$(FIGS_INTRO)/TSintro.pdf: $(IntroPlotScripts)/TSintro.py
	python $< $(TSINTRO)/fine $(TSINTRO)/coarse $(TSINTRO)/quantized $@

## ALL           : All of the targets that Rules.mk is responsible for
ALL = $(FIGS_INTRO)/TSintro.pdf

# Local Variables:
# mode: makefile
# End:

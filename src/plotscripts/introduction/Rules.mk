# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
TSINTRO = $(SYNTHETIC_DATA)/TSintro
IntroPlotScripts = $(PLOTSCRIPTS)/introduction
FIGS_INTRO = $(BUILD)/figs/introduction

# Note: The file lorenz.flag is touched after the files fine, coarse,
# and quantized get built.
$(FIGS_INTRO)/TSintro.pdf: $(IntroPlotScripts)/ts_intro.py $(SYNTHETIC_DATA)/lorenz.flag
	mkdir -p $(FIGS_INTRO)
	python $< $(TSINTRO)/fine $(TSINTRO)/coarse $(TSINTRO)/quantized $@

$(FIGS_INTRO)/Statesintro.pdf: $(IntroPlotScripts)/stateplot.py $(SYNTHETIC_DATA)/states
	mkdir -p $(FIGS_INTRO)
	python $<  --data_dir $(SYNTHETIC_DATA) --base_name state --fig_path $@

# Local Variables:
# mode: makefile
# End:

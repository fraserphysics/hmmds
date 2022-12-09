# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and ApneaPlotScripts are defined.  ROOT is the top directory
# of the hmmds project and ApneaPlotScripts is the directory where
# this file is located.

ApneaFigDir = $(ROOT)/build/figs/apnea
ApneaDerivedData = $(ROOT)/build/derived_data/apnea
EXPERT =  $(ROOT)/raw_data/apnea/summary_of_training
LPHR = $(ApneaDerivedData)/Lphr
RESPIRE = $(ApneaDerivedData)/Respire

APNEA_FIGS = $(addprefix $(ApneaFigDir)/, $(addsuffix .pdf, a03erA a03erN a03HR ApneaNLD sgram))

APNEA_TS_PLOTS = $(ApneaPlotScripts)/apnea_ts_plots.py
$(ApneaFigDir)/a03erA.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er.pickle
	python $< --data_dir $(ApneaDerivedData)  $@

$(ApneaFigDir)/a03erN.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er.pickle
	python $< --data_dir $(ApneaDerivedData)  $@

$(ApneaFigDir)/a03HR.pdf: $(APNEA_TS_PLOTS) $(LPHR)/flag
	python $< --data_dir $(ApneaDerivedData)  $@

$(ApneaFigDir)/ApneaNLD.pdf: $(APNEA_TS_PLOTS) $(LPHR)/a01.lphr $(LPHR)/flag
	python $< --data_dir $(ApneaDerivedData)  $@

$(ApneaFigDir)/sgram.pdf:  $(ApneaPlotScripts)/spectrogram.py $(RESPIRE)/flag
	python $< --time_window 40 220  --frequency_window 1 30 --name a11 \
$(LPHR) $(RESPIRE) $(EXPERT) $@

# Local Variables:
# mode: makefile
# End:

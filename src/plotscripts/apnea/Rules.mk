# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and ApneaPlotScripts are defined.  ROOT is the top directory
# of the hmmds project and ApneaPlotScripts is the directory where
# this file is located.

ApneaFigDir = $(ROOT)/build/figs/apnea
ApneaDerivedData = $(ROOT)/build/derived_data/apnea
EXPERT =  $(ROOT)/raw_data/apnea/summary_of_training
RESPIRE = $(ApneaDerivedData)/Respire

APNEA_FIGS = $(addprefix $(ApneaFigDir)/, $(addsuffix .pdf, a03erA a03erN a03HR ApneaNLD sgram ))

APNEA_TS_PLOTS = $(ApneaPlotScripts)/apnea_ts_plots.py
PLOT_COMMAND = 	mkdir -p $(@D); python $(APNEA_TS_PLOTS) --root $(ROOT) --heart_rate_path_format $(ROOT)/build/derived_data/ECG/{0}_self_AR3/heart_rate

$(ApneaFigDir)/a03erA.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er.pickle
	$(PLOT_COMMAND)   $@

$(ApneaFigDir)/a03erN.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er.pickle
	$(PLOT_COMMAND)   $@

$(ApneaFigDir)/a03HR.pdf: $(APNEA_TS_PLOTS)
	$(PLOT_COMMAND)   $@

$(ApneaFigDir)/ApneaNLD.pdf: $(APNEA_TS_PLOTS)
	$(PLOT_COMMAND)    $@

$(ApneaFigDir)/sgram.pdf:  $(ApneaPlotScripts)/spectrogram.py $(ApneaDerivedData)/a11.sgram
	mkdir -p $(@D)
	python $< --root $(ROOT) --time_window 20 150  --frequency_window 5 25 --record_name a11 \
$(ApneaDerivedData)/a11.sgram $(EXPERT) $@



# Local Variables:
# mode: makefile
# End:

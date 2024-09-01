# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and ApneaPlotScripts are defined.  ROOT is the top directory
# of the hmmds project and ApneaPlotScripts is the directory where
# this file is located.

ApneaFigDir = $(ROOT)/build/figs/apnea
ApneaDerivedData = $(ROOT)/build/derived_data/apnea
EXPERT =  $(ROOT)/raw_data/apnea/summary_of_training
RESPIRE = $(ApneaDerivedData)/Respire
APNEA_PLOTSCRIPTS = $(ROOT)/src/plotscripts/apnea

ALL_SELVES = ${ROOT}/build/derived_data/ECG/all_selves

APNEA_FIGS = $(addprefix $(ApneaFigDir)/, $(addsuffix .pdf, a03erA a03erN a03HR ApneaNLD sgram ))

APNEA_TS_PLOTS = $(APNEA_PLOTSCRIPTS)/apnea_ts_plots.py
PLOT_COMMAND = 	mkdir -p $(@D); python $(APNEA_TS_PLOTS) --root $(ROOT) --heart_rate_path_format $(ROOT)/build/derived_data/ECG/{0}_self_AR3/heart_rate

$(ApneaFigDir)/a03erA.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er.pkl
	$(PLOT_COMMAND)   $@

$(ApneaFigDir)/a03erN.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er.pkl
	$(PLOT_COMMAND)   $@

$(ApneaFigDir)/a03HR.pdf: $(APNEA_TS_PLOTS)
	$(PLOT_COMMAND)   $@

$(ApneaFigDir)/ApneaNLD.pdf: $(APNEA_TS_PLOTS) $(ALL_SELVES)
	$(PLOT_COMMAND)    $@

$(ApneaFigDir)/sgram.pdf:  $(APNEA_PLOTSCRIPTS)/spectrogram.py $(ApneaDerivedData)/a11.sgram
	mkdir -p $(@D)
	python $< --root $(ROOT) --time_window 20 150  --frequency_window 5 25 --record_name a11 \
$(ApneaDerivedData)/a11.sgram $(EXPERT) $@

$(APNEA_FIG_DIR)/errors_vs_%.pdf: $(APNEA_PLOTSCRIPTS)/comparison_plot.py $(DERIVED_APNEA_DATA)/errors_vs_%.pkl
	mkdir -p $(@D)
	python $^ $@

$(APNEA_FIG_DIR)/viz.pdf: $(ApneaCode)/model_viz.py $(BEST)
	mkdir -p $(@D)
	python $^ $@

$(APNEA_FIG_DIR)/threshold.pdf: $(APNEA_PLOTSCRIPTS)/survey_threshold.py $(BEST)
	mkdir -p $(@D)
	python $< --root $(ROOT) --expert_override $(ROOT)/raw_data/apnea/summary_of_training --heart_rate_path_format $(ECG_DERIVED)/{0}_self_AR3/heart_rate --records $(TRAIN_NAMES) --thresholds -0.3 0.3 21 \
 $(BEST) $@ > $(DERIVED_APNEA_DATA)/threshold.txt


# Local Variables:
# mode: makefile
# End:

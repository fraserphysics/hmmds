# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and ApneaPlotScripts are defined.  ROOT is the top directory
# of the hmmds project and ApneaPlotScripts is the directory where
# this file is located.

ApneaFigDir = $(ROOT)/build/figs/apnea
ApneaDerivedData = $(ROOT)/build/derived_data/apnea
EXPERT =  $(ROOT)/raw_data/apnea/summary_of_training
LPHR = $(ApneaDerivedData)/Lphr
RESPIRE = $(ApneaDerivedData)/Respire

APNEA_FIGS = $(addprefix $(ApneaFigDir)/, $(addsuffix .pdf, a03erA a03erN a03HR ApneaNLD sgram pass1)) $(ApneaFigDir)/lda_flag
#lda_flag is for LDA1.pdf and LDA2.pdf

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

$(ApneaFigDir)/lda_flag: $(ApneaPlotScripts)/lda.py  $(ApneaDerivedData)/Respire/lda_data
	python $< --apnea_data_dir $(ApneaDerivedData) $(ApneaFigDir)/LDA1 $(ApneaFigDir)/LDA2
	touch $@

$(ApneaFigDir)/pass1.pdf: $(ApneaPlotScripts)/pass1.py  $(ApneaDerivedData)/pass1_report.pickle
	python $^  $@


# Local Variables:
# mode: makefile
# End:

# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and ApneaPlotScripts are defined.  ROOT is the top directory
# of the hmmds project and ApneaPlotScripts is the directory where
# this file is located.

ApneaFigDir = $(ROOT)/figs/apnea
ApneaDerivedData = $(ROOT)/derived_data/apnea
LPHR_DIR = $(ApneaDerivedData)/low_pass_heart_rate

APNEA_FIGS = $(addprefix $(ApneaFigDir)/, $(addsuffix .pdf, a03erA a03erN a03HR ApneaNLD))

APNEA_TS_PLOTS = $(ApneaPlotScripts)/apnea_ts_plots.py
$(ApneaFigDir)/a03erA.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er_seg
	python $< --data_dir $(ApneaDerivedData)  $@

$(ApneaFigDir)/a03erN.pdf: $(APNEA_TS_PLOTS) $(ApneaDerivedData)/a03er_seg
	python $< --data_dir $(ApneaDerivedData)  $@

$(ApneaFigDir)/a03HR.pdf: $(APNEA_TS_PLOTS) $(LPHR_DIR)/a03
	python $< --data_dir $(ApneaDerivedData)  $@

$(ApneaFigDir)/ApneaNLD.pdf: $(APNEA_TS_PLOTS) $(LPHR_DIR)/a01 $(LPHR_DIR)/a12
	python $< --data_dir $(ApneaDerivedData)  $@

# Local Variables:
# mode: makefile
# End:

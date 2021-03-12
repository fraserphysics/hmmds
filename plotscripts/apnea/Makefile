Figs = ../../figs/apnea
DerivedData = ../../derived_data/apnea
LPHR_DIR = $(DerivedData)/low_pass_heart_rate

plots: $(addprefix $(Figs)/, $(addsuffix .pdf, a03erA a03erN a03HR ApneaNLD))

$(Figs)/a03erA.pdf: apnea_ts_plots.py $(DerivedData)/a03er_seg
	python apnea_ts_plots.py --data_dir $(DerivedData)  $@

$(Figs)/a03erN.pdf: apnea_ts_plots.py $(DerivedData)/a03er_seg
	python apnea_ts_plots.py --data_dir $(DerivedData)  $@

$(Figs)/a03HR.pdf: apnea_ts_plots.py $(LPHR_DIR)/a03
	python apnea_ts_plots.py --data_dir $(DerivedData)  $@

$(Figs)/ApneaNLD.pdf: apnea_ts_plots.py $(LPHR_DIR)/a01 $(LPHR_DIR)/a12
	python apnea_ts_plots.py --data_dir $(DerivedData)  $@

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo plots: $(plots)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

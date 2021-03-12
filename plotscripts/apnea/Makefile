DerivedDataApnea = ../../derived_data/apnea
a03er_seg = $(DerivedDataApnea)/a03er_seg
LPHR = $(addprefix $(DerivedDataApnea)/low_pass_heart_rate/, a03 a01 a12)
PLOTS = a03erA_plot a03erN_plot a03HR_plot ApneaNLD
# It takes 2 seconds to make all 4 plots.  Because make doesn't handle
# multiple targets from a single rule well, I will write separate
# rules for 4 targets
a03erA.pdf:
	python apnea_ts_plots.py $(a03er_seg) $(LPHR) $(PLOTS)
#python apnea_ts_plots.py a03er_seg a03_lphr a01_lphr a12_lphr a03erA_plot a03erN_plot a03HR_plot ApneaNLD

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo LPHR: $(LPHR)
	@echo PLOTS: $(PLOTS)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

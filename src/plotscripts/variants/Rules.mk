# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.

SYNTHETIC_DATA = $(BUILD)/derived_data/synthetic
SGOData =  $(SYNTHETIC_DATA)/SGO_sim
SGOFlag =  $(SYNTHETIC_DATA)/SGO
VARGData =  $(SYNTHETIC_DATA)/vstates
# vstates is a sentinel for $(addprefix $(SYNTHETIC_DATA)/varg_state, 0 1 2 3 4 5 6 7 8 9 10 11)
VariantPlotScripts = $(PLOTSCRIPTS)/variants
STATEPLOT = $(PLOTSCRIPTS)/introduction/stateplot.py
FIGS_Variants = $(BUILD)/figs/variants

# Rule for SGO_b, SGO_c, and SGO_d
$(FIGS_Variants)/SGO_bcd: $(VariantPlotScripts)/scalar_gaussian.py $(SGOFlag)
	mkdir -p $(@D)
	python $< $(SGOData) $(FIGS_Variants)
	touch $@

$(FIGS_Variants)/VARGstates.pdf: $(STATEPLOT) $(VARGData)
	mkdir -p  $(@D)
	python $< --data_dir $(SYNTHETIC_DATA) --base_name varg_state --fig_path $@

# Local Variables:
# mode: makefile
# End:

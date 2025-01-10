# Rules.mk: For making figures of ecgs and their analysis.  This file
# can be included by a makefile anywhere as long as the following are
# defined:

# ROOT         : The root of this project
# BUILD        : Directory tree for derived results
# ECG_DERIVED  : Holds pickled data
# PICKLED_ECG  : Directory of pickled ecg data from wfdb

ECG_FIG_DIR = $(BUILD)/figs/ecg
ECG_PLOTSCRIPTS = $(ROOT)/src/plotscripts/ecg
# This file is in the ECG_PLOTSCRIPTS directory

################################################################################
$(ECG_FIG_DIR)/ecg2hr.pdf: $(ECG_PLOTSCRIPTS)/ecg2hr.py $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(ECG_DERIVED)/a01_self_AR3 $(PICKLED_ECG)/a01 $@

$(ECG_FIG_DIR)/a03a10b03c02.pdf: $(ECG_PLOTSCRIPTS)/four_ecgs.py $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< --records a03 a10 b03 c02 $(PICKLED_ECG) $@

$(ECG_FIG_DIR)/%_states_71.pdf: $(ECG_PLOTSCRIPTS)/ecg_states_fig.py $(ECG_DERIVED)/%/states/a01 $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)/a01 $(ECG_DERIVED)/$*/states/a01 71.1 71.14 $@

$(ECG_FIG_DIR)/%_states_70.pdf: $(ECG_PLOTSCRIPTS)/ecg_states_fig.py $(ECG_DERIVED)/%/states/a01 $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)/a01 $(ECG_DERIVED)/$*/states/a01 70.785 70.825 $@

$(ECG_FIG_DIR)/a01a19c02.pdf:  $(ECG_PLOTSCRIPTS)/a01a19c02.py $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02 $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(PICKLED_ECG) $(ECG_DERIVED)/a01_trained_AR3/states/c02 300.0 300.05 $@

$(ECG_FIG_DIR)/train_log.pdf: $(ECG_PLOTSCRIPTS)/train_characteristic.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained
	mkdir -p $(@D)
	python $< $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained.log $@

$(ECG_FIG_DIR)/like_a14_x07.pdf: $(ECG_PLOTSCRIPTS)/plot_like.py $(ECG_DERIVED)/a01_trained_AR3/all_states_likelihood_heart_rate $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(PICKLED_ECG) $(ECG_DERIVED)/a01_trained_AR3/likelihood/ 300.16 300.19 a14 x07 $@

$(ECG_FIG_DIR)/simulated.pdf: $(ECG_PLOTSCRIPTS)/plot_simulation.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained
	mkdir -p $(@D)
	python $^ 1000 $@

$(ECG_FIG_DIR)/elgendi.pdf: $(ECG_PLOTSCRIPTS)/elgendi.py $(ECG_DERIVED)/a03_ElgendiRtimes $(ECG_DERIVED)/a03_self_AR3/states $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< --ecg_models $(ECG_DERIVED) --ecg_dir $(PICKLED_ECG) --rtimes $(ECG_DERIVED)/a03_ElgendiRtimes $@

$(ECG_FIG_DIR)/constant_a03.pdf: $(ECG_PLOTSCRIPTS)/constant_a03.py $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)/a03 $@

$(ECG_FIG_DIR)/a03_states_56.pdf:  $(ECG_PLOTSCRIPTS)/ecg_states_fig.py $(ECG_DERIVED)/a03_self_AR3/states $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)/a03 $(ECG_DERIVED)/a03_self_AR3/states 56.25 56.4 $@

$(ECG_FIG_DIR)/a01c02_states.pdf:  $(ECG_PLOTSCRIPTS)/a01c02_states.py $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02 $(PICKLED_ECG)/flag
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)  $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02 300.0 300.05 $@

# Local Variables:
# mode: makefile
# End:

# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

# I get the raw ecg data from files in the ecgs directory.  The rule
# for $(ECG)/%_self_AR3/heart_rate makes the best heart rate
# estimates.

PICKLED_ECG = $(ROOT)/build/derived_data/apnea/ecgs
ECG_DERIVED = ${ROOT}/build/derived_data/ECG
ECG_FIGS = $(BUILD)/figs/ecg
ECG_PLOTSCRIPTS = $(ROOT)/src/plotscripts/ecg
ECG_TeX = $(BUILD)/TeX/ecg
ECGCode = $(HMMDS)/applications/apnea/ECG
# This file is in the ECGCode directory

# Data built elsewhere
PHYSIONET_WFDB = $(ROOT)/raw_data/apnea/apnea-ecg-database

XNAMES = x01 x02 x03 x04 x05 x06 x07 x08 x09 x10 \
x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 \
x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 \
x31 x32 x33 x34 x35

ANAMES = a01 a02 a03 a04 a05 a06 a07 a08 a09 a10 \
a11 a12 a13 a14 a15 a16 a17 a18 a19 a20

BNAMES = b01 b02 b03 b04

# b05 is a mess

CNAMES = c01 c02 c03 c04 c05 c06 c07 c08 c09 c10

NAMES = $(ANAMES) $(BNAMES) $(CNAMES) $(XNAMES)

KEEPERS = initial masked_trained unmasked_trained states likelihood heart_rate

.PRECIOUS: $(foreach X,$(ECG_DIRS),$(foreach Y,$(KEEPERS),$X/$Y))

$(PICKLED_ECG)/flag: $(ECGCode)/wfdb2pickle_ecg.py
	mkdir -p $(@D)
	python $< $(PHYSIONET_WFDB) $(@D) $(NAMES)
	ls $(@D)/x35  # Weak test of completion
	touch $@

# Because Gnu make doesn't support pattern rules with two patterns, I
# have the following section of almost repeated pattern rules:

#################### Block for a01_trained_AR3 ##############################
$(ECG_DERIVED)/a01_trained_AR3/states/%: $(ECGCode)/ecg_decode.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG_DERIVED)/a01_trained_AR3/likelihood/%: $(ECGCode)/ecg_likelihood.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG_DERIVED)/a01_trained_AR3/heart_rate/%: $(ECGCode)/states2hr.py $(ECG_DERIVED)/a01_trained_AR3/states/%
	mkdir -p  $(@D)
	python $<  $(ECG_DERIVED)/a01_trained_AR3/states/$* $@

$(ECG_DERIVED)/a01_trained_AR3/all_states_likelihood_heart_rate: $(foreach X, states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG_DERIVED)/a01_trained_AR3/, $X/$Y)))
	touch $@

#################### Rules to make a01_trained_AR3/unmasked_trained ############
$(ECG_DERIVED)/a01_trained_AR3/initial: $(ECGCode)/model_init.py $(PICKLED_ECG)/flag
	mkdir -p  $(@D)
	python $(ECGCode)/model_init.py --root ${ROOT} --records a01 --tag_ecg \
--ecg_alpha_beta 1.0e3 1.0e2 --noise_parameters 1.0e6 1.0e8 1.0e-50 \
--before_after_slow 18 30 10 --AR_order 3 masked_dict $@

$(ECG_DERIVED)/a01_trained_AR3/masked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/a01_trained_AR3/initial
	python $< --records a01 --type segmented --iterations 5 \
$(ECG_DERIVED)/a01_trained_AR3/initial $@ >  $(ECG_DERIVED)/a01_trained_AR3/masked.log

$(ECG_DERIVED)/a01_trained_AR3/unmasked_hmm: $(ECGCode)/declass.py $(ECG_DERIVED)/a01_trained_AR3/masked_trained
	python $^ $@

$(ECG_DERIVED)/a01_trained_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_hmm
	python $< --records a01 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

## Pattern rules for models of other records starting from a01_trained_AR3/unmasked_hmm ###
$(ECG_DERIVED)/%_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained
	mkdir -p $(@D)
	python $< --records $* --type segmented --iterations 10 $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained $@ >  $@.log
$(ECG_DERIVED)/%_self_AR3/states: $(ECGCode)/ecg_decode.py $(ECG_DERIVED)/%_self_AR3/unmasked_trained
	python $^ --root $(ROOT) $* $@
$(ECG_DERIVED)/%_self_AR3/likelihood: $(ECGCode)/ecg_likelihood.py $(ECG_DERIVED)/%_self_AR3/unmasked_trained
	python $^ --root $(ROOT) $* $@
$(ECG_DERIVED)/%_self_AR3/heart_rate: $(ECGCode)/states2hr.py $(ECG_DERIVED)/%_self_AR3/states $(ECG_DERIVED)/%_self_AR3/likelihood
	python $< --root $(ROOT) --r_state 35 --likelihood $(ECG_DERIVED)/$*_self_AR3/likelihood 0.02 $(ECG_DERIVED)/$*_self_AR3/states $@

# Models with alpha and beta to force variance 0.008 for all normal
# states.  That keeps noisy ecg between beats from being assigned to R
# part of cycle.  This code is not used 2024-05-02.
$(ECG_DERIVED)/%_self_AR3/fixed_sigma_initial: $(ECGCode)/model_init.py
	mkdir -p  $(@D)
	python $(ECGCode)/model_init.py --root ${ROOT} --records $* --tag_ecg \
--ecg_alpha_beta 1.0e8 0.008e8 --noise_parameters 1.0e6 1.0e8 1.0e-50 \
--before_after_slow 18 30 10 --AR_order 3 masked_dict $@

$(ECG_DERIVED)/%_self_AR3/fixed_sigma_masked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/%_self_AR3/fixed_sigma_initial
	python $< --records $* --type segmented --iterations 5 \
$(ECG_DERIVED)/$*_self_AR3/fixed_sigma_initial $@ >  $(ECG_DERIVED)/$*_self_AR3/masked.log

$(ECG_DERIVED)/%_self_AR3/fixed_sigma_unmasked_hmm: $(ECGCode)/declass.py $(ECG_DERIVED)/%_self_AR3/fixed_sigma_masked_trained
	python $^ $@

$(ECG_DERIVED)/xxx_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/xxx_self_AR3/fixed_sigma_unmasked_hmm
	mkdir -p $(@D)
	python $< --records xxx --type segmented --iterations 10 $(ECG_DERIVED)/xxx_self_AR3/fixed_sigma_unmasked_hmm $@ >  $@.log


##########Special rules for records with arrhythmia#############################
$(ECG_DERIVED)/x11_self_AR3/heart_rate: $(ECGCode)/states2hr.py $(ECG_DERIVED)/x11_self_AR3/states $(ECG_DERIVED)/x11_self_AR3/likelihood
	python $< --root $(ROOT) --r_state 35 --likelihood $(ECG_DERIVED)/x11_self_AR3/likelihood 0.01 \
--ecg_peaks x11 0.03 --hr_dips 55 $(ECG_DERIVED)/x11_self_AR3/states $@

$(ECG_DERIVED)/x13_self_AR3/heart_rate: $(ECGCode)/states2hr.py $(ECG_DERIVED)/x13_self_AR3/states $(ECG_DERIVED)/x13_self_AR3/likelihood
	python $< --root $(ROOT) --r_state 35 --likelihood $(ECG_DERIVED)/x13_self_AR3/likelihood 0.02 \
--ecg_peaks x13 -0.05 $(ECG_DERIVED)/x13_self_AR3/states $@

$(ECG_DERIVED)/x19_self_AR3/heart_rate: $(ECGCode)/states2hr.py $(ECG_DERIVED)/x19_self_AR3/states $(ECG_DERIVED)/x19_self_AR3/likelihood
	python $< --root $(ROOT) --r_state 35 --likelihood $(ECG_DERIVED)/x19_self_AR3/likelihood 0.1 \
$(ECG_DERIVED)/x19_self_AR3/states $@

# Special model for x26.  alpha and beta force variance 0.016 for all normal states.  Otherwise noisy ecg between beats gets assigned to R part of cycle.
$(ECG_DERIVED)/x26_self_AR3/initial: $(ECGCode)/model_init.py
	mkdir -p  $(@D)
	python $(ECGCode)/model_init.py --root ${ROOT} --records x26 --tag_ecg \
--ecg_alpha_beta 1.0e8 0.016e8 --noise_parameters 1.0e6 1.0e8 1.0e-50 \
--before_after_slow 18 30 10 --AR_order 3 masked_dict $@

$(ECG_DERIVED)/x26_self_AR3/masked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/x26_self_AR3/initial
	python $< --records x26 --type segmented --iterations 5 \
$(ECG_DERIVED)/x26_self_AR3/initial $@ >  $(ECG_DERIVED)/x26_self_AR3/masked.log
$(ECG_DERIVED)/x26_self_AR3/unmasked_hmm: $(ECGCode)/declass.py $(ECG_DERIVED)/x26_self_AR3/masked_trained
	python $^ $@
$(ECG_DERIVED)/x26_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/x26_self_AR3/unmasked_hmm
	mkdir -p $(@D)
	python $< --records x26 --type segmented --iterations 10 $(ECG_DERIVED)/x26_self_AR3/unmasked_hmm $@ >  $@.log

$(ECG_DERIVED)/x26_self_AR3/heart_rate: $(ECGCode)/states2hr.py $(ECG_DERIVED)/x26_self_AR3/states $(ECG_DERIVED)/x26_self_AR3/likelihood
	python $< --root $(ROOT) --r_state 35 --likelihood $(ECG_DERIVED)/x26_self_AR3/likelihood 0.03 \
--ecg_peaks x26 -0.03 --hr_dips 48 $(ECG_DERIVED)/x26_self_AR3/states $@

##### Special rules for records that don't work with models trained on a01 ########
$(ECG_DERIVED)/a12_self_AR3/initial: $(ECGCode)/model_init.py
	mkdir -p  $(@D)
	python $(ECGCode)/model_init.py --root ${ROOT} --records a12 --tag_ecg \
--ecg_alpha_beta 1.0e8 0.016e8 --noise_parameters 1.0e6 1.0e8 1.0e-50 \
--before_after_slow 18 30 10 --AR_order 3 masked_dict $@

$(ECG_DERIVED)/a12_self_AR3/masked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/a12_self_AR3/initial
	python $<  --records a12 --type segmented --iterations 5 \
$(ECG_DERIVED)/a12_self_AR3/initial $@ >  $(ECG_DERIVED)/a12_self_AR3/masked.log

$(ECG_DERIVED)/a12_self_AR3/unmasked_hmm: $(ECGCode)/declass.py $(ECG_DERIVED)/a12_self_AR3/masked_trained
	python $^ $@

$(ECG_DERIVED)/a12_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/a12_self_AR3/unmasked_hmm
	python $< --records a12 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

$(ECG_DERIVED)/a12_self_AR3/heart_rate: $(ECGCode)/states2hr.py $(ECG_DERIVED)/a12_self_AR3/states $(ECG_DERIVED)/a12_self_AR3/likelihood
	python $< --root $(ROOT) --r_state 35 --likelihood $(ECG_DERIVED)/a12_self_AR3/likelihood 0.03 \
--ecg_peaks a12 -0.03 --hr_dips 48 $(ECG_DERIVED)/a12_self_AR3/states $@

$(ECG_DERIVED)/c07_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG_DERIVED)/c10_self_AR3/unmasked_trained
	mkdir -p $(@D)
	python $< --records c07 --type segmented --iterations 20 $(ECG_DERIVED)/c10_self_AR3/unmasked_trained $@ >  $@.log

################################################################################
$(ECG_FIGS)/%_states_71.pdf: $(ECG_PLOTSCRIPTS)/ecg_states_fig.py $(ECG_DERIVED)/%/states/a01
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)/a01 $(ECG_DERIVED)/$*/states/a01 71.1 71.14 $@

$(ECG_FIGS)/%_states_70.pdf: $(ECG_PLOTSCRIPTS)/ecg_states_fig.py $(ECG_DERIVED)/%/states/a01
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)/a01 $(ECG_DERIVED)/$*/states/a01 70.785 70.825 $@

$(ECG_FIGS)/a01a19c02.pdf:  $(ECG_PLOTSCRIPTS)/a01a19c02.py $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02
	mkdir -p $(@D)
	python $< $(PICKLED_ECG) $(ECG_DERIVED)/a01_trained_AR3/states/c02 300.0 300.05 $@

$(ECG_FIGS)/train_log.pdf: $(ECG_PLOTSCRIPTS)/train_characteristic.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained
	mkdir -p $(@D)
	python $< $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained.log $@

$(ECG_FIGS)/like_a14_x07.pdf: $(ECG_PLOTSCRIPTS)/plot_like.py $(ECG_DERIVED)/a01_trained_AR3/all_states_likelihood_heart_rate
	mkdir -p $(@D)
	python $< $(PICKLED_ECG) $(ECG_DERIVED)/a01_trained_AR3/likelihood/ 300.16 300.19 a14 x07 $@

$(ECG_FIGS)/simulated.pdf: $(ECG_PLOTSCRIPTS)/plot_simulation.py $(ECG_DERIVED)/a01_trained_AR3/unmasked_trained
	mkdir -p $(@D)
	python $^ 1000 $@

$(ECG_DERIVED)/table.tex: table.py $(ECG_DERIVED)/a01_trained_AR3/all_states_likelihood_heart_rate
	python $< $(ECG_DERIVED)/a01_trained_AR3/ $@

$(ECG_DERIVED)/self_table.tex: self_table.py $(ECG_DERIVED)/all_selves
	python $< $(ECG_DERIVED) $@

ECG_FIGLIST = $(addsuffix .pdf, $(addprefix $(ECG_FIGS)/, \
a01_trained_AR3_states_70 a01_trained_AR3_states_71 a01a19c02 train_log \
like_a14_x07 simulated))

$(ECG_TeX)/ecg.pdf: ecg.tex $(ECG_FIGLIST) $(ECG_DERIVED)/table.tex $(ECG_DERIVED)/self_table.tex
	mkdir -p $(@D)
	export TEXINPUTS=$(abspath $(BUILD))//:; \
        pdflatex --output-directory=$(@D) $< ; pdflatex --output-directory=$(@D) $<

################################################################################

$(ECG_FIGS)/elgendi.pdf: $(ECG_PLOTSCRIPTS)/elgendi.py
	mkdir -p $(@D)
	python $< $@

$(ECG_FIGS)/constant_a03.pdf: $(ECG_PLOTSCRIPTS)/constant_a03.py  $(PICKLED_ECG)a03
	mkdir -p $(@D)
	python $^ $@

$(ECG_FIGS)/a03_states_56.pdf:  $(ECG_PLOTSCRIPTS)/ecg_states_fig.py $(ECG_DERIVED)/a03_self_AR3/states
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)/a03 $(ECG_DERIVED)/a03_self_AR3/states 56.25 56.4 $@

$(ECG_FIGS)/a01c02_states.pdf:  $(ECG_PLOTSCRIPTS)/a01c02_states.py $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02
	mkdir -p $(@D)
	python $< $(PICKLED_ECG)  $(ECG_DERIVED)/a01_trained_AR3/states/a01 $(ECG_DERIVED)/a01_trained_AR3/states/c02 300.0 300.05 $@

$(ECG_FIGS)/%.pdf : %.fig
	mkdir -p $(@D)
	fig2dev -L pdftex -F $< $@
$(ECG_FIGS)/%.pdf_t: %.fig $(ECG_FIGS)/%.pdf
	mkdir -p $(@D)
	fig2dev -L pdftex_t -p $(ECG_FIGS)/$*.pdf $< $@

DS23FIGS = $(addsuffix .pdf, $(addprefix $(ECG_FIGS)/, elgendi constant_a03 \
a03_states_56 a01c02_states ecg_hmm simulated)) $(ECG_FIGS)/ecg_hmm.pdf_t

$(ECG_TeX)/ds23.pdf: ds23.tex $(DS23FIGS)
	export TEXINPUTS=$(ABSROOT)//:; pdflatex --output-directory=$(@D) $<; pdflatex --output-directory=$(@D) $<

################################################################################
all_selves = $(foreach X, unmasked_trained states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG_DERIVED)/$Y_self_AR3/, $X)))
$(ECG_DERIVED)/all_selves: $(all_selves)
	touch $@

# $ time make ../../../../../build/derived_data/ECG/all_selves
# real	183m36.882s
# user	1390m39.927s
# sys	38m11.636s

# explore.py indicates that the model for x04 inserts extra beats.

# Local Variables:
# mode: makefile
# End:

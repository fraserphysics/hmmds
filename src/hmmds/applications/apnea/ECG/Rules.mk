# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

# Data built elsewhere
RTIMES = $(ROOT)/raw_data/Rtimes/
ECG = ${ROOT}/build/derived_data/ECG
ECGCode = $(HMMDS)/applications/apnea/ECG
# This file is in the ECGCode directory

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

# Because Gnu make doesn't support pattern rules with two patterns, I
# have the following section of almost repeated pattern rules:

#################### Block for a01_trained_AR3 ##############################
$(ECG)/a01_trained_AR3/states/%: $(ECGCode)/ecg_decode.py $(ECG)/a01_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/a01_trained_AR3/likelihood/%: $(ECGCode)/ecg_likelihood.py $(ECG)/a01_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/a01_trained_AR3/heart_rate/%: $(ECGCode)/states2hr.py $(ECG)/a01_trained_AR3/states/%
	mkdir -p  $(@D)
	python $<  $(ECG)/a01_trained_AR3/states/$* $@

$(ECG)/a01_trained_AR3/all_states_likelihood_heart_rate: $(foreach X, states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG)/a01_trained_AR3/, $X/$Y)))
	touch $@

#################### Rules to make a01_trained_AR3/unmasked_trained ############
$(ECG)/a01_trained_AR3/initial: model_init.py
	mkdir -p  $(@D)
	python $(ECGCode)/model_init.py --root ${ROOT} --records a01 --tag_ecg \
--ecg_alpha_beta 1.0e3 1.0e2 --noise_parameters 1.0e6 1.0e8 1.0e-50 \
--before_after_slow 18 30 10 --AR_order 3 masked_dict $@

$(ECG)/a01_trained_AR3/masked_trained: $(ECGCode)/train.py $(ECG)/a01_trained_AR3/initial
	python $< --records a01 --type segmented --iterations 5 \
$(ECG)/a01_trained_AR3/initial $@ >  $(ECG)/a01_trained_AR3/masked.log

$(ECG)/a01_trained_AR3/unmasked_hmm: $(ECGCode)/declass.py $(ECG)/a01_trained_AR3/masked_trained
	python $^ $@

$(ECG)/a01_trained_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG)/a01_trained_AR3/unmasked_hmm
	python $< --records a01 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

## Pattern rules for models of other records starting from a01_trained_AR3/unmasked_hmm ###
$(ECG)/%_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG)/a01_trained_AR3/unmasked_trained
	mkdir -p $(@D)
	python $< --records $* --type segmented --iterations 10 $(ECG)/a01_trained_AR3/unmasked_trained $@ >  $@.log
$(ECG)/%_self_AR3/states: $(ECGCode)/ecg_decode.py $(ECG)/%_self_AR3/unmasked_trained
	python $^ $* $@
$(ECG)/%_self_AR3/likelihood: $(ECGCode)/ecg_likelihood.py $(ECG)/%_self_AR3/unmasked_trained
	python $^ $* $@
$(ECG)/%_self_AR3/heart_rate: $(ECGCode)/states2hr.py $(ECG)/%_self_AR3/states $(ECG)/%_self_AR3/likelihood
	python $< --r_state 35 --likelihood $(ECG)/$*_self_AR3/likelihood --censor 0.02 $(ECG)/$*_self_AR3/states $@

#################### Special rules for models trained on a single record ########
$(ECG)/a12_self_AR3/initial: model_init.py
	mkdir -p  $(@D)
	python $(ECGCode)/model_init.py --root ${ROOT} --records a12 \
--peak_scale -.2 --tag_ecg --ecg_alpha_beta 1.0e3 1.0e2 \
--noise_parameters 1.0e6 1.0e8 1.0e-50 --before_after_slow 18 30 2 \
--AR_order 3 masked_dict $@

$(ECG)/a12_self_AR3/masked_trained: $(ECGCode)/train.py $(ECG)/a12_self_AR3/initial
	python $<  --records a12 --type segmented --iterations 5 \
$(ECG)/a12_self_AR3/initial $@ >  $(ECG)/a12_self_AR3/masked.log

$(ECG)/a12_self_AR3/unmasked_hmm: $(ECGCode)/declass.py $(ECG)/a12_self_AR3/masked_trained
	python $^ $@

$(ECG)/a12_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG)/a12_self_AR3/unmasked_hmm
	python $< --records a12 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

$(ECG)/c07_self_AR3/initial: model_init.py
	mkdir -p  $(@D)
	python $(ECGCode)/model_init.py --root ${ROOT} --records c07 \
--peak_scale .3 --tag_ecg --ecg_alpha_beta 1.0e3 1.0e2 \
--noise_parameters 1.0e6 1.0e8 1.0e-50 --before_after_slow 18 30 10 \
--AR_order 3 masked_dict $@

$(ECG)/c07_self_AR3/masked_trained: $(ECGCode)/train.py $(ECG)/c07_self_AR3/initial
	python $<  --records c07 --type segmented --iterations 5 \
$(ECG)/c07_self_AR3/initial $@ >  $(ECG)/c07_self_AR3/masked.log

$(ECG)/c07_self_AR3/unmasked_hmm: $(ECGCode)/declass.py $(ECG)/c07_self_AR3/masked_trained
	python $^ $@

$(ECG)/c07_self_AR3/unmasked_trained: $(ECGCode)/train.py $(ECG)/c07_self_AR3/unmasked_hmm
	python $< --records c07 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

################################################################################
all_selves = $(foreach X, unmasked_trained states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG)/$Y_self_AR3/, $X)))
$(ECG)/all_selves: $(all_selves)
	touch $@

# Local Variables:
# mode: makefile
# End:

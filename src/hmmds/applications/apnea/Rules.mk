# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

# Data built elsewhere
RTIMES = /mnt/cheap/home/andy/Rtimes/
RAW_DATA = /mnt/precious/home/andy_nix/projects/dshmm/raw_data
EXPERT =  $(RAW_DATA)/apnea/summary_of_training
OLDA03ER = /mnt/precious/home/andy_nix/projects/dshmm/build/derived_data/apnea/a03er.pickle

ApneaDerivedData = $(ROOT)/build/derived_data/apnea
ApneaCode = $(HMMDS)/applications/apnea
# This file is in the ApneaCode directory

LPHR = $(ApneaDerivedData)/Lphr
RESPIRE = $(ApneaDerivedData)/Respire
# HMMs go in ${MODELS}
MODELS = ${ROOT}/build/derived_data/apnea/models
ECG = $(MODELS)/ECG

# I made the Rtimes files using the script wfdb2Rtimes.py in my
# project wfdb which imports PhysioNet's wfdb using its own shell.nix
# that is incompatible with qt.  The script uses a qrs detector from
# ecgdetectors.

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

$(ApneaDerivedData)/a03er.pickle: $(OLDA03ER)
	cp $< $@

$(RTIMES)/%.rtimes:
	$(error Use wfdb code to extract $@ from PhysioNet database)

# Flag signifies all *.lphr files have been built
$(LPHR)/flag : $(addsuffix .lphr, $(addprefix $(LPHR)/, $(NAMES)))
	touch $@

$(RTIMES)/flag: $(addsuffix .rtimes, $(addprefix $(RTIMES)/, $(NAMES)))
	touch $@

# Pattern rule for making $(LPHR)/foo.lphr from $(RTIMES)/foo.rtimes
$(LPHR)/%.lphr: $(ApneaCode)/rtimes2hr.py $(RTIMES)/%.rtimes
	mkdir -p $(LPHR)
	python $< $(RTIMES) $(LPHR) $*

$(RESPIRE)/flag: $(ApneaCode)/respire.py $(EXPERT) $(RTIMES)/flag
	mkdir -p $(RESPIRE)
	python $< $(EXPERT) $(RTIMES) $(RESPIRE)
	touch $@

# The trained files are expensive to build.  Don't delete them
.PRECIOUS: ${MODELS}/initial_% ${MODELS}/p1model_%

# Rule for initial models
${MODELS}/initial_%: $(ApneaCode)/model_init.py $(ApneaCode)/utilities.py $(ApneaCode)/observation.py $(RESPIRE)/flag $(LPHR)/flag
	mkdir -p ${MODELS}
	python $(ApneaCode)/model_init.py --root ${ROOT} $* $@

# Rule for pass 1 models
${MODELS}/p1model_%: $(ApneaCode)/apnea_train.py ${MODELS}/initial_%
	python $< --iterations 5 --root ${ROOT} $* $(word 2,$^) $@

################## Begin ECG Targets #########################################

# Each ECG_DIR has a unique rule for making its "unmasked_trained" file
ECG_NAMES = a01_trained_AR3 diverse_trained_AR3
ECG_DIRS =  $(addprefix $(ECG)/, $(ECG_NAMES))
KEEPERS = initial masked_trained unmasked_trained

# The trained files are expensive to build.  Don't delete them
.PRECIOUS: $(foreach X,$(ECG_DIRS),$(foreach Y,$(KEEPERS),$X/$Y))

# Because Gnu make doesn't support pattern rules with two patterns, I
# have the following section of almost repeated pattern rules:

#################### Block for a01_trained_AR3 ##############################
$(ECG)/a01_trained_AR3/states/%: $(ApneaCode)/ecg_decode.py $(ECG)/a01_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/a01_trained_AR3/likelihood/%: $(ApneaCode)/ecg_likelihood.py $(ECG)/a01_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/a01_trained_AR3/heart_rate/%: $(ApneaCode)/states2hr.py $(ECG)/a01_trained_AR3/states/%
	mkdir -p  $(@D)
	python $<  $(ECG)/a01_trained_AR3/states/$* $@

$(ECG)/a01_trained_AR3/all_states_likelihood_heart_rate: $(foreach X, states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG)/a01_trained_AR3/, $X/$Y)))
	touch $@

#################### Block for diverse_trained_AR3 ###########################
$(ECG)/diverse_trained_AR3/states/%: $(ApneaCode)/ecg_decode.py $(ECG)/diverse_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/diverse_trained_AR3/likelihood/%: $(ApneaCode)/ecg_likelihood.py $(ECG)/diverse_trained_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/diverse_trained_AR3/heart_rate/%: $(ApneaCode)/states2hr.py $(ECG)/diverse_trained_AR3/states/%
	mkdir -p  $(@D)
	python $<  $(ECG)/diverse_trained_AR3/states/$* $@

$(ECG)/diverse_trained_AR3/all_states_likelihood_heart_rate: $(foreach X, states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG)/diverse_trained_AR3/, $X/$Y)))
	touch $@


#################### Block for implausible_AR3 ###########################
$(ECG)/implausible_AR3/states/%: $(ApneaCode)/ecg_decode.py $(ECG)/implausible_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/implausible_AR3/likelihood/%: $(ApneaCode)/ecg_likelihood.py $(ECG)/implausible_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/implausible_AR3/heart_rate/%: $(ApneaCode)/states2hr.py $(ECG)/implausible_AR3/states/%
	mkdir -p  $(@D)
	python $<  $(ECG)/implausible_AR3/states/$* $@

$(ECG)/implausible_AR3/all_states_likelihood_heart_rate: $(foreach X, states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG)/implausible_AR3/, $X/$Y)))
	touch $@


#################### Block for funny_AR3 ###########################
$(ECG)/funny_AR3/states/%: $(ApneaCode)/ecg_decode.py $(ECG)/funny_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/funny_AR3/likelihood/%: $(ApneaCode)/ecg_likelihood.py $(ECG)/funny_AR3/unmasked_trained
	mkdir -p  $(@D)
	python $^ $* $@
$(ECG)/funny_AR3/heart_rate/%: $(ApneaCode)/states2hr.py $(ECG)/funny_AR3/states/%
	mkdir -p  $(@D)
	python $<  $(ECG)/funny_AR3/states/$* $@

$(ECG)/funny_AR3/all_states_likelihood_heart_rate: $(foreach X, states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG)/funny_AR3/, $X/$Y)))
	touch $@


#################### Rules to make a01_trained_AR3/unmasked_trained ############
$(ECG)/a01_trained_AR3/initial: model_init.py
	mkdir -p  $(@D)
	python $(ApneaCode)/model_init.py --root ${ROOT} --records a01 --tag_ecg \
--ecg_alpha_beta 1.0e3 1.0e2 --noise_parameters 1.0e6 1.0e8 1.0e-50 \
--before_after_slow 18 30 10 --AR_order 3 masked_dict $@

# 2023-04-14 note: The model made with --noise_parameters 1.0e2 1.0e4
# 1.0e-3 says the a03 data is impossible in 5 places.  I think the
# weak prior (alpha and beta) let the noise state have a variance that
# is too small.

# --tag_ecg arg for masked_trained not necessary because initial
# --carries its own args
$(ECG)/a01_trained_AR3/masked_trained: $(ApneaCode)/train.py $(ECG)/a01_trained_AR3/initial
	python $<  --records a01 --type segmented --iterations 5 \
$(ECG)/a01_trained_AR3/initial $@ >  $(ECG)/a01_trained_AR3/masked.log

$(ECG)/a01_trained_AR3/unmasked_hmm: $(ApneaCode)/declass.py $(ECG)/a01_trained_AR3/masked_trained
	python $^ $@

$(ECG)/a01_trained_AR3/unmasked_trained: $(ApneaCode)/train.py $(ECG)/a01_trained_AR3/unmasked_hmm
	python $< --records a01 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log



#################### Rules to make diverse_trained_AR3/unmasked_trained ########

# I looked at ecg signals and put signals that didn't look typical to
# me in the list FUNNY.  When I first ran train.py with --type
# diverse, the program rejected the records in IMPLAUSIBLE because
# they don't have 15 minute segments of plausible data.  I think the
# signals in those files are bigger.

# a01 1.5 mV

# a03 2.8mV
# a07 3mV
# a08 wandering base up to 7 mV
# a15 3mV
# a16 more than 4mV
# a19 3mV
# x01 3.5mV
# x16 Base wanders by 3mV
# x19 almost 4mV
# x20 3.5mV
# x21 3mV
# x27 3.5 mV
# x30 3mV

# Diverse gets trained on a01 a02 a03 a04 a05 a06 a09 a11 a13 a14 a17
# a18 a20 b01 b03 b04 c01 c03 c04 c05 c06 c07 c08 c09 c10 x02 x03 x04
# x08 x09 x10 x12 x13 x14 x15 x16 x17 x18 x19 x22 x23 x24 x25 x26 x28
# x29 x31 x32 x35
FUNNY =       a10 a12 b02 c02 x05 x06 x07 x11 x33 x34
IMPLAUSIBLE = a03 a07 a08 a15 a16 a19 x01 x16 x19 x20 x21 x27 x30
DIVERSE_RECORDS = $(filter-out $(FUNNY) $(IMPLAUSIBLE), $(NAMES))

$(ECG)/diverse_trained_AR3/unmasked_trained: $(ApneaCode)/train.py $(ECG)/a01_trained_AR3/unmasked_hmm  $(ECG)/a01_trained_AR3/all_states_likelihood_heart_rate
	mkdir -p $(@D)
	python $< --records $(DIVERSE_RECORDS) --type diverse --iterations 20 $(ECG)/a01_trained_AR3/unmasked_hmm $@ >  $@.log

# I base the next two directories on the trained diverse model because
# it is more flexible that the model trained on a01.

#################### Rule to make implausible_AR3/unmasked_trained ########

$(ECG)/implausible_AR3/unmasked_trained: $(ApneaCode)/train.py $(ECG)/diverse_trained_AR3/unmasked_trained  $(ECG)/diverse_trained_AR3/all_states_likelihood_heart_rate
	mkdir -p $(@D)
	python $< --records $(IMPLAUSIBLE) --type implausible --iterations 20 $(ECG)/diverse_trained_AR3/unmasked_trained $@ >  $@.log

#################### Rule to make funny_AR3/unmasked_trained ########

$(ECG)/funny_AR3/unmasked_trained: $(ApneaCode)/train.py $(ECG)/diverse_trained_AR3/unmasked_trained  $(ECG)/diverse_trained_AR3/all_states_likelihood_heart_rate
	mkdir -p $(@D)
	python $< --records $(FUNNY) --type implausible --iterations 20 $(ECG)/diverse_trained_AR3/unmasked_trained $@ >  $@.log

#################### Pattern rules for models trained on a single record #####
$(ECG)/%_self_AR3/unmasked_trained: $(ApneaCode)/train.py $(ECG)/a01_trained_AR3/unmasked_trained
	mkdir -p $(@D)
	python $< --records $* --type segmented --iterations 10 $(ECG)/a01_trained_AR3/unmasked_trained $@ >  $@.log
$(ECG)/%_self_AR3/states: $(ApneaCode)/ecg_decode.py $(ECG)/%_self_AR3/unmasked_trained
	python $^ $* $@
$(ECG)/%_self_AR3/likelihood: $(ApneaCode)/ecg_likelihood.py $(ECG)/%_self_AR3/unmasked_trained
	python $^ $* $@
$(ECG)/%_self_AR3/heart_rate: $(ApneaCode)/states2hr.py $(ECG)/%_self_AR3/states
	python $<  --r_state 35 $(ECG)/$*_self_AR3/states $@

#################### Special rules for models trained on a single record ########
$(ECG)/a12_self_AR3/initial: model_init.py
	mkdir -p  $(@D)
	python $(ApneaCode)/model_init.py --root ${ROOT} --records a12 \
--peak_scale -.2 --tag_ecg --ecg_alpha_beta 1.0e3 1.0e2 \
--noise_parameters 1.0e6 1.0e8 1.0e-50 --before_after_slow 18 30 10 \
--AR_order 3 masked_dict $@

$(ECG)/a12_self_AR3/masked_trained: $(ApneaCode)/train.py $(ECG)/a12_self_AR3/initial
	python $<  --records a12 --type segmented --iterations 5 \
$(ECG)/a12_self_AR3/initial $@ >  $(ECG)/a12_self_AR3/masked.log

$(ECG)/a12_self_AR3/unmasked_hmm: $(ApneaCode)/declass.py $(ECG)/a12_self_AR3/masked_trained
	python $^ $@

$(ECG)/a12_self_AR3/unmasked_trained: $(ApneaCode)/train.py $(ECG)/a12_self_AR3/unmasked_hmm
	python $< --records a12 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

$(ECG)/c07_self_AR3/initial: model_init.py
	mkdir -p  $(@D)
	python $(ApneaCode)/model_init.py --root ${ROOT} --records c07 \
--peak_scale .3 --tag_ecg --ecg_alpha_beta 1.0e3 1.0e2 \
--noise_parameters 1.0e6 1.0e8 1.0e-50 --before_after_slow 18 30 10 \
--AR_order 3 masked_dict $@

$(ECG)/c07_self_AR3/masked_trained: $(ApneaCode)/train.py $(ECG)/c07_self_AR3/initial
	python $<  --records c07 --type segmented --iterations 5 \
$(ECG)/c07_self_AR3/initial $@ >  $(ECG)/c07_self_AR3/masked.log

$(ECG)/c07_self_AR3/unmasked_hmm: $(ApneaCode)/declass.py $(ECG)/c07_self_AR3/masked_trained
	python $^ $@

$(ECG)/c07_self_AR3/unmasked_trained: $(ApneaCode)/train.py $(ECG)/c07_self_AR3/unmasked_hmm
	python $< --records c07 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

################################################################################
all_selves = $(foreach X, unmasked_trained states likelihood heart_rate , $(foreach Y, $(NAMES), $(addprefix $(ECG)/$Y_self_AR3/, $X)))
$(ECG)/all_selves: $(all_selves)
	touch $@

# real	341m5.299s
# user	5348m59.570s
# sys	75m9.104s

#####################Test higher AR order######################################
ORDER=9
$(ECG)/a12_self_AR$(ORDER)/initial: model_init.py
	mkdir -p  $(@D)
	python $(ApneaCode)/model_init.py --root ${ROOT} --records a12 \
--peak_scale -.2 --tag_ecg --ecg_alpha_beta 1.0e3 1.0e2 \
--noise_parameters 1.0e6 1.0e8 1.0e-50 --before_after_slow 18 30 10 \
--AR_order $(ORDER) masked_dict $@

$(ECG)/a12_self_AR$(ORDER)/masked_trained: $(ApneaCode)/train.py $(ECG)/a12_self_AR$(ORDER)/initial
	python $<  --records a12 --type segmented --iterations 5 \
$(ECG)/a12_self_AR$(ORDER)/initial $@ >  $(ECG)/a12_self_AR$(ORDER)/masked.log

$(ECG)/a12_self_AR$(ORDER)/unmasked_hmm: $(ApneaCode)/declass.py $(ECG)/a12_self_AR$(ORDER)/masked_trained
	python $^ $@

$(ECG)/a12_self_AR$(ORDER)/unmasked_trained: $(ApneaCode)/train.py $(ECG)/a12_self_AR$(ORDER)/unmasked_hmm
	python $< --records a12 --type segmented --iterations 20 $(@D)/unmasked_hmm $@ >  $@.log

$(ECG)/%_self_AR$(ORDER)/states: $(ApneaCode)/ecg_decode.py $(ECG)/%_self_AR$(ORDER)/unmasked_trained
	python $^ $* $@
	ln -s $(@D) $(ECG)/x36_self_AR3  # Bogus for self_explore.py
$(ECG)/%_self_AR$(ORDER)/likelihood: $(ApneaCode)/ecg_likelihood.py $(ECG)/%_self_AR$(ORDER)/unmasked_trained
	python $^ $* $@
$(ECG)/%_self_AR$(ORDER)/heart_rate: $(ApneaCode)/states2hr.py $(ECG)/%_self_AR$(ORDER)/states
	python $<  --r_state 35 $(ECG)/$*_self_AR3/states $@

################## End ECG Targets #########################################

# Use p1model_A4 and p1model_C2 to create file with lines like: x24 # Low
# stat= 1.454 llr= -0.603 R= 1.755.  For each line, calculate
# Low/Medium/High using stat, low_stat, and high_stat.  40 minutes on cathcart
${ApneaDerivedData}/pass1_report.pickle: $(ApneaCode)/pass1.py ${MODELS}/p1model_A4 ${MODELS}/p1model_C2
	python $(ApneaCode)/pass1.py --root ${ROOT} --Amodel p1model_A4 --BCmodel p1model_C2

# Local Variables:
# mode: makefile
# End:

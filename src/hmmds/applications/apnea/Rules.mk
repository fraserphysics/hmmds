# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

DERIVED_APNEA_DATA = $(BUILD)/derived_data/apnea
APNEA_FIG_DIR = $(BUILD)/figs/apnea
ApneaCode = $(HMMDS)/applications/apnea
# This file is in the ApneaCode directory

# Data built elsewhere
EXPERT =  $(ROOT)/raw_data/apnea/summary_of_training
PHYSIONET_WFDB = $(ROOT)/raw_data/apnea/apnea-ecg-database

MODELS = ${ROOT}/build/derived_data/apnea/models
ECG = $(MODELS)/ECG

# See hmmds/applications/apnea/ECG/Makefile for making files like
# build/derived_data/ECG/a01_self_AR3/heart_rate

# Data files in a03er are shorter than claimed in a03er.hea
$(DERIVED_APNEA_DATA)/a03er.pkl: $(ApneaCode)/extract_wfdb.py
	mkdir -p $(@D)
	python $< --shorten 204 $(PHYSIONET_WFDB) a03er $@

$(DERIVED_APNEA_DATA)/a11.sgram: $(ApneaCode)/spectrogram.py
	python $< --root $(ROOT) --model_sample_frequency 120 --fft_width 256 \
--band_pass_center $(RC) --band_pass_width $(RW) --low_pass_period $(LPP) a11 $@

XNAMES = x01 x02 x03 x04 x05 x06 x07 x08 x09 x10 \
x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 \
x21 x22 x23 x24 x25 x26 x27 x28 x29 x30 \
x31 x32 x33 x34 x35

ANAMES = a01 a02 a03 a04 a05 a06 a07 a08 a09 a10 \
a11 a12 a13 a14 a15 a16 a17 a18 a19 a20

BNAMES = b01 b02 b03 b04
# b05 is a mess

CNAMES = c01 c02 c03 c05 c07 c08 c09 c10
# c04 has arrhythmia, and c06 is the same as c05

ALL_NAMES = $(ANAMES) $(BNAMES) $(CNAMES) $(XNAMES)

TRAIN_NAMES = $(ANAMES) $(BNAMES) $(CNAMES)


# AutoRegressive order
AR = 8
# Model sample frequency cpm
FS = 4
# Low Pass Period seconds
LPP = 65
# Respiration Center frequency cpm
RC = 12
# Respiration Width cpm
RW = 3.6
# Filter for Respiration Smoothing in cpm.
RS = 0.47
# Detection threshold
THRESHOLD = 0.7

$(MODELS)/%_init: $(ApneaCode)/make_init.py $(ApneaCode)/model_init.py
	mkdir -p  $(@D)
	python $< --root $(ROOT) multi_state $* $@

BEST = $(MODELS)/ar$(AR)fs$(FS)lpp$(LPP)rc$(RC)rw$(RW)rs$(RS)_masked

$(MODELS)/%_masked: $(ApneaCode)/apnea_train.py $(MODELS)/%_init
	python $< --root $(ROOT) --records $(TRAIN_NAMES) --iterations 40 $(MODELS)/$*_init $@

$(DERIVED_APNEA_DATA)/pass2.out: $(ApneaCode)/pass2.py $(BEST)
	python $^ $@  --root $(ROOT) --records $(TRAIN_NAMES) --threshold $(THRESHOLD)

$(DERIVED_APNEA_DATA)/test_pass2.out: $(ApneaCode)/pass2.py $(BEST)
	python $^ $@  --root $(ROOT) --records $(XNAMES) --threshold $(THRESHOLD)

$(DERIVED_APNEA_DATA)/score.tex: $(ApneaCode)/score.py $(DERIVED_APNEA_DATA)/pass2.out
	python $^ $@  --tex  --root $(ROOT)

$(DERIVED_APNEA_DATA)/test_score.tex: $(ApneaCode)/score.py $(DERIVED_APNEA_DATA)/test_pass2.out
	python $^ $@ $(XNAMES)  --root $(ROOT) --expert raw_data/apnea/event-2-answers --tex

########################Build data for hand_opt.pdf#############################

# Sensitivity to AutoRegressive order
ARs = 5 6 7 8 9 10 11 12
AR_MODELS = $(addsuffix fs$(FS)lpp$(LPP)rc$(RC)rw$(RW)rs$(RS)_masked, $(addprefix ar, $(ARs)))

$(DERIVED_APNEA_DATA)/errors_vs_ar.pkl: $(ApneaCode)/compare_models.py $(addprefix $(MODELS)/, $(AR_MODELS))
	$(COMPARE) --models $(AR_MODELS) --parameters $(ARs) \
--parameter_name "AR order" $@ > $(DERIVED_APNEA_DATA)/errors_vs_ar.txt

# Sensitivity to Sample Frequency (7 not allowed)
FSs = 2 3 4 5 6 8
FS_MODELS = $(addsuffix lpp$(LPP)rc$(RC)rw$(RW)rs$(RS)_masked, \
    $(addprefix ar$(AR)fs, $(FSs)))

$(DERIVED_APNEA_DATA)/errors_vs_fs.pkl: $(ApneaCode)/compare_models.py $(addprefix $(MODELS)/, $(FS_MODELS))
	$(COMPARE) --models $(FS_MODELS) --parameters $(FSs) \
--parameter_name "Sample Frequency" $@ > $(DERIVED_APNEA_DATA)/errors_vs_fs.txt

# Sensitivity to Low Pass Period in seconds.
LPPs = 55 60 65 68 70 72 75 85
LPP_MODELS = $(addsuffix rc$(RC)rw$(RW)rs$(RS)_masked, $(addprefix \
    ar$(AR)fs$(FS)lpp, \
    $(LPPs)))

$(DERIVED_APNEA_DATA)/errors_vs_lpp.pkl: $(ApneaCode)/compare_models.py $(addprefix $(MODELS)/, $(LPP_MODELS))
	$(COMPARE) --models $(LPP_MODELS) --parameters $(LPPs) \
--parameter_name "Low Pass Period" $@ > $(DERIVED_APNEA_DATA)/errors_vs_lpp.txt

# Sensitivity to Respiration Center frequency in cpm
RCs = 11 11.5 11.8 12 12.2 12.5 13 14
RC_MODELS = $(addsuffix rw$(RW)rs$(RS)_masked, $(addprefix \
    ar$(AR)fs$(FS)lpp$(LPP)rc, \
    $(RCs)))

$(DERIVED_APNEA_DATA)/errors_vs_rc.pkl: $(ApneaCode)/compare_models.py $(addprefix $(MODELS)/, $(RC_MODELS))
	$(COMPARE) --models $(RC_MODELS) --parameters $(RCs) \
--parameter_name "Center Frequency" $@ > $(DERIVED_APNEA_DATA)/errors_vs_rc.txt

# Sensitivity to Respiration width frequency in cpm
RWs = 3.3 3.4 3.5 3.6 3.7 4.0
RW_MODELS = $(addsuffix rs$(RS)_masked, $(addprefix \
    ar$(AR)fs$(FS)lpp$(LPP)rc$(RC)rw, \
    $(RWs)))

$(DERIVED_APNEA_DATA)/errors_vs_rw.pkl: $(ApneaCode)/compare_models.py $(addprefix $(MODELS)/, $(RW_MODELS))
	$(COMPARE) --models $(RW_MODELS) --parameters $(RWs) \
--parameter_name "Frequency Width" $@ > $(DERIVED_APNEA_DATA)/errors_vs_rw.txt

# Sensitivity to Respiration Smoothing frequency in cpm
RSs = .46 .465 .47 .475 .48
RS_MODELS = $(addsuffix _masked, $(addprefix \
    ar$(AR)fs$(FS)lpp$(LPP)rc$(RC)rw$(RW)rs, \
    $(RSs)))

$(DERIVED_APNEA_DATA)/errors_vs_rs.pkl: $(ApneaCode)/compare_models.py $(addprefix $(MODELS)/, $(RS_MODELS))
	$(COMPARE) --models $(RS_MODELS) --parameters $(RSs) \
    --parameter_name "Respiration Smoothing Filter" \
    $@ > $(DERIVED_APNEA_DATA)/errors_vs_rs.txt

COMPARE = python $(ApneaCode)/compare_models.py --root $(ROOT) --records $(TRAIN_NAMES) --threshold $(THRESHOLD)

$(BUILD)/TeX/book/apnea_values.tex: $(ApneaCode)/tex_values.py
	mkdir -p $(@D)
	python $< --command_line ArOrder $(AR) ModelSampleFrequency $(FS) LowPassPeriod $(LPP) RespirationCenterFrequency $(RC) RespirationFilterWidth $(RW) RespirationSmoothing $(RS) -- $@
# Local Variables:
# mode: makefile
# End:

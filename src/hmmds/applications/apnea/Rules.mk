# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

ApneaDerivedData = $(ROOT)/build/derived_data/apnea
ApneaCode = $(HMMDS)/applications/apnea
# This file is in the ApneaCode directory
RAW_DATA = $(ROOT)/raw_data
EXPERT =  $(ROOT)/raw_data/apnea/summary_of_training
LPHR = $(ApneaDerivedData)/Lphr
RESPIRE = $(ApneaDerivedData)/Respire
RTIMES = $(ApneaDerivedData)/Rtimes
# HMMs go in ${MODELS}
MODELS = ${ROOT}/build/derived_data/apnea/models

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

$(ApneaDerivedData)/a03er.pickle:
	$(error Use wfdb code to extract $@ from PhysioNet database)

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
.PRECIOUS: $(addprefix ${MODELS}/, modelA2 model_Low model_Medium model_High initial_Low)

# Rule for initial models: initial_A2, initial_C1, initial_High, initial_Medium, initial_Low
${MODELS}/initial_%: $(ApneaCode)/model_init.py $(ApneaCode)/utilities.py $(ApneaCode)/observation.py $(RESPIRE)/flag $(LPHR)/flag
	mkdir -p ${MODELS}
	python $(ApneaCode)/model_init.py --root ${ROOT} $* $@

# Rule for pass 1 models
${MODELS}/p1model_%: $(ApneaCode)/apnea_train.py ${MODELS}/initial_%
	python $< --iterations 5 --root ${ROOT} $* $(word 2,$^) $@

${MODELS}/model_outlier: $(ApneaCode)/apnea_train.py ${MODELS}/initial_outlier
	python $(ApneaCode)/apnea_train.py --iterations 1 --root ${ROOT} outlier ${MODELS}/initial_outlier $@

# Use p1model_A4 and p1model_C2 to create file with lines like: x24 # Low
# stat= 1.454 llr= -0.603 R= 1.755.  For each line, calculate
# Low/Medium/High using stat, low_stat, and high_stat.  40 minutes on cathcart
${ApneaDerivedData}/pass1_report.pickle: $(ApneaCode)/pass1.py ${MODELS}/p1model_A4 ${MODELS}/p1model_C2
	python $(ApneaCode)/pass1.py --root ${ROOT} --Amodel p1model_A4 --BCmodel p1model_C2

# Local Variables:
# mode: makefile
# End:

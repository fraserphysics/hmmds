# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

# Data built elsewhere
RTIMES = /mnt/cheap/home/andy/Rtimes/
RAW_DATA = /mnt/precious/home/andy_nix/projects/dshmm/raw_data
EXPERT =  $(RAW_DATA)/apnea/summary_of_training

ApneaDerivedData = $(ROOT)/build/derived_data/apnea
ApneaCode = $(HMMDS)/applications/apnea
# This file is in the ApneaCode directory

MODELS = ${ROOT}/build/derived_data/apnea/models
ECG = $(MODELS)/ECG

# See hmmds/applications/apnea/ECG/Makefile for making files like
# build/derived_data/ECG/a01_self_AR3/heart_rate

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

CNAMES = c01 c02 c03 c05 c07 c08 c09 c10
# c04 has arrhythmia, and c06 is the same as c05

ALL_NAMES = $(ANAMES) $(BNAMES) $(CNAMES) $(XNAMES)

TRAIN_NAMES = $(ANAMES) $(BNAMES) $(CNAMES)

# Local Variables:
# mode: makefile
# End:

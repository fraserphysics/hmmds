######################ECG#########################################
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../../..
ABSROOT = $(abspath $(ROOT))
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

# Default target.  Note: ecg.pdf depends indirectly on
# $(ECG_DERIVED)/all_selves, and building $(ECG_DERIVED)/all_selves
# builds the heart rate files that I use for apnea detection.

ECG_DEFAULT = $(BUILD)/TeX/ecg/ecg.pdf
.PHONY : evince
evince: $(ECG_DEFAULT)
	evince $< &

include Rules.mk

.PHONY : ds23
ds23: $(ECG_TeX)/ds23.pdf
	evince $< &

As = a01 a03 a05 a07 a09 a11 a13 a15 a17 a19 a02 a04 a06 a08 a10 a12	\
a14 a16 a18 a20
Bs = b01  b02  b03  b04
Cs = c01  c02  c03  c04  c05  c06  c07  c08  c09  c10
Xs = x01 x04 x07 x10 x13 x16 x19 x22 x25 x28 x31 x34 x02 x05 x08 x11	\
x14 x17 x20 x23 x26 x29 x32 x35 x03 x06 x09 x12 x15 x18 x21 x24 x27	\
x30 x33

## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" .

## variables     : Print selected variables.
.PHONY : variables
variables:
	@echo ECG_FIGS: $(ECG_FIGS)
	@echo ECG_DERIVED: $(ECG_DERIVED)

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

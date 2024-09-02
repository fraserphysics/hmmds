######################ECG#########################################
# Several paths and parameters are shared in instances of utilities.Common.

ROOT = ../../../../..
ABSROOT = $(abspath $(ROOT))
HMMDS = $(ROOT)/src/hmmds
BUILD = $(ROOT)/build

include Rules.mk

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

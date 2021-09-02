# Rules.mk: This file can be included by a makefile anywhere as long
# as SYNTHETIC_DATA and SYNTHETIC_CODE are defined.  SYNTHETIC_CODE is
# the directory of source files and where this file is located.

$(SYNTHETIC_DATA)/lorenz.flag: $(SYNTHETIC_CODE)/lorenz.py
	mkdir -p $(SYNTHETIC_DATA)/TSintro
	python $< --n_samples 2050 --quantfile $(SYNTHETIC_DATA)/lorenz.4 --xyzfile $(SYNTHETIC_DATA)/lorenz.xyz  --TSintro $(SYNTHETIC_DATA)/TSintro
	touch $@

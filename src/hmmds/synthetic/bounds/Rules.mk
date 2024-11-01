# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the root of this project and
# BUILD is where derived results go.

BOUNDS_DATA = $(BUILD)/derived_data/synthetic/bounds
BOUNDS_CODE = $(ROOT)/src/hmmds/synthetic/bounds
# BOUNDS_CODE is this directory

$(BOUNDS_DATA)/data_h_view:  $(BOUNDS_CODE)/h_cli.py
	mkdir -p $(BOUNDS_DATA)
	python $< $@

$(BOUNDS_DATA)/toy_h: $(BOUNDS_CODE)/toy_h.py
	mkdir -p $(BOUNDS_DATA)
	python $< --n_t 200 --t_steps .02 .53 .125 --log_steps -3.5 -5.6 -.5 $@

$(BOUNDS_DATA)/benettin: $(BOUNDS_CODE)/benettin.py
	mkdir -p $(BOUNDS_DATA)
	python $< --n_times=400 --n_runs=1000 --dev_state 1e-5 --grid_size 1e-3 --n_relax 50 $@

$(BOUNDS_DATA)/like_lor: $(BOUNDS_CODE)/like_lor.py
	mkdir -p $(BOUNDS_DATA)
	python $< $@

$(BUILD)/TeX/book/toy_values.tex: $(BOUNDS_CODE)/toy_values.py $(addprefix $(BOUNDS_DATA)/, data_h_view toy_h benettin like_lor)
	mkdir -p $(@D)
	python $^ $@

# Local Variables:
# mode: makefile
# End:

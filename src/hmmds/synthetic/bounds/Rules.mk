# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT and BUILD are defined.  ROOT is the root of this project and
# BUILD is where derived results go.

BOUNDS_DATA = $(BUILD)/derived_data/synthetic/bounds
BOUNDS_CODE = $(ROOT)/src/hmmds/synthetic/bounds
# BOUNDS_CODE is this directory

$(BOUNDS_DATA)/data_h_view:  $(BOUNDS_CODE)/h_cli.py
	mkdir -p $(@D)
	python $< $@

$(BOUNDS_DATA)/toy_h: $(BOUNDS_CODE)/toy_h.py
	mkdir -p $(@D)
	python $< --n_t 200 --t_steps .02 .53 .125 --log_steps -3.5 -5.6 -.5 $@

$(BOUNDS_DATA)/benettin: $(BOUNDS_CODE)/benettin.py
	mkdir -p $(@D)
	python $< --atol 1e-8 --time_step 0.15 --t_run 300 --t_estimate 10000 $@

$(BOUNDS_DATA)/like_lor: $(BOUNDS_CODE)/like_lor.py
	mkdir -p $(@D)
	python $< --log_resolution 3 -5.6 -0.5 --n_train 10000000 --n_test 10000 $@
#	python $< --log_resolution 3 -1.6 -0.5 --n_train 100000 --n_test 10000 $@

$(BOUNDS_DATA)/particle_1k: $(BOUNDS_CODE)/particle.py $(BOUNDS_CODE)/filter.py
	mkdir -p $(@D)
	python $< --margin 0.01 --s_augment 5e-4 --resample 10000 4000 --r_threshold 1e-3 $@

$(BOUNDS_DATA)/ddays: $(BOUNDS_CODE)/particle.py $(BOUNDS_CODE)/filter.py
	mkdir -p $(@D)
	python $< --n_y 100 $@

$(BUILD)/TeX/book/toy_values.tex: $(BOUNDS_CODE)/toy_values.py $(addprefix $(BOUNDS_DATA)/, data_h_view toy_h benettin like_lor)
	mkdir -p $(@D)
	python $^ $@

# Local Variables:
# mode: makefile
# End:

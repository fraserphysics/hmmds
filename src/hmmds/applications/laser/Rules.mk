SHELL=bash
# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

LASER_DATA = $(BUILD)/derived_data/synthetic
LASER_CODE = $(HMMDS)/applications/laser
# LASER_CODE is this directory
LP5DAT = $(ROOT)/raw_data/LP5.DAT

# This is to ensure Cython code is compiled
LORENZ_SDE = $(HMMDS)/synthetic/filter/lorenz_sde.flag

# Define POWELL250 as an abbreviation
POWELL250 = $(LASER_DATA)/ekf_powell250.parameters

# Define the action to build a plot_data target
PLOT_DATA = python $< --method skip --plot_data $@ --laser_data $(LP5DAT) $(word 2, $^)
# $< is the first dependency
# $@ is the target
# $(word 2, $^) is the second dependency

$(LASER_DATA)/%.plot_data: $(LASER_CODE)/optimize_ekf.py $(LASER_DATA)/%.parameters $(LORENZ_SDE)
	$(PLOT_DATA)

$(LASER_DATA)/pf_%.plot_data: $(LASER_CODE)/optimize_particle.py $(LASER_DATA)/%.parameters $(LORENZ_SDE)
	$(PLOT_DATA)

$(LASER_DATA)/pf_ekf250.plot_data: $(LASER_CODE)/optimize_particle.py $(POWELL250) $(LORENZ_SDE)
	$(PLOT_DATA)

# Define the action to build a plot_data target with override
OVERRIDE = python $< --method skip --plot_data $@ --laser_data $(LP5DAT) \
--override_parameters $(word 3, $^) $(word 2, $^)

$(LASER_DATA)/pf_opt_noise.plot_data: $(LASER_CODE)/optimize_particle.py $(POWELL250) \
$(LASER_DATA)/pf_powell500.parameters $(LORENZ_SDE)
	$(OVERRIDE)

$(LASER_DATA)/pf_hand_noise.plot_data: $(LASER_CODE)/optimize_particle.py $(POWELL250) \
$(LASER_DATA)/pf_hand_noise.parameters $(LORENZ_SDE)
	$(OVERRIDE)

$(LASER_DATA)/gui.plot_data: $(LASER_CODE)/optimize_ekf.py $(LASER_CODE)/explore.txt $(LORENZ_SDE)
	mkdir -p $(LASER_DATA)
	python $< --method skip --plot_data $@ --parameter_type GUI_out --laser_data $(LP5DAT) \
$(word 2, $^)

$(LASER_DATA)/%.particle_data: $(LASER_CODE)/optimize_particle.py $(LASER_DATA)/%.parameters $(LORENZ_SDE)
	python $< --method skip --n_particles 300 --laser_data $(LP5DAT) --plot_data $@ \
$(LASER_DATA)/$*.parameters

$(LASER_DATA)/pf_hand_noise.parameters: $(LASER_DATA)/pf_powell500.parameters
	sed -e "s/^state_noise.*/state_noise 0.2/" -e "s/^observation_noise.*/observation_noise 1.0/" < $< > $@

$(LASER_DATA)/pf_powell500.parameters: $(LASER_CODE)/optimize_particle.py $(POWELL250) $(LORENZ_SDE)
	python $< --length 175 --n_particles 500 --method Powell --laser_data $(LP5DAT) $(POWELL250) $@

# 5 hours -17.050910252989638 intermediate result, but final is -139.2018738220304?
$(LASER_DATA)/particle.parameters: $(LASER_CODE)/optimize_particle.py $(POWELL250) $(LORENZ_SDE)
	python $< --length 175 --n_particles 100 --method Powell --laser_data $(LP5DAT) $(POWELL250) $@

# Takes about 22 minutes
$(LASER_DATA)/ekf_powell2876.parameters: $(LASER_CODE)/optimize_ekf.py $(POWELL250) $(LORENZ_SDE)
	python $< --parameter_type parameter --laser_data $(LP5DAT) --length 2876 \
--method Powell $(POWELL250) $@

$(LASER_DATA)/l2.parameters: $(LASER_CODE)/optimize_ekf.py $(LASER_CODE)/explore.txt $(LORENZ_SDE)
	mkdir -p $(LASER_DATA)
	python $< --parameter_type GUI_out --laser_data $(LP5DAT) --length 250 \
--method Powell --objective_function l2 $(word 2, $^) $@

$(POWELL250): $(LASER_CODE)/optimize_ekf.py $(LASER_DATA)/l2.parameters $(LORENZ_SDE)
	mkdir -p $(LASER_DATA)
	python $< --parameter_type parameter --laser_data $(LP5DAT) --length 250 \
--method Powell --objective_function likelihood $(word 2, $^) $@

# $(LASER_CODE)/explore.txt is under version control
$(LASER_CODE)/explore.txt: $(LASER_CODE)/explore.py
	echo Adjust sliders for period 5 orbit, match laser data, and press "write" button
	cd $(LASER_CODE); python $(LASER_CODE)/explore.py

$(LASER_DATA)/LaserLikeOptTS: $(LASER_CODE)/figure_data.py $(LASER_DATA)/l2.parameters
	python $< --parameters $(word 2, $^) --laser_data $(LP5DAT) --LaserLP5 $@

# Pattern rule for LaserLP5 LaserLogLike LaserStates LaserForecast LaserHist
$(LASER_DATA)/Laser%: $(LASER_CODE)/figure_data.py $(POWELL250)
	python $< --parameters $(POWELL250) --laser_data $(LP5DAT) --Laser$* $@

# Local Variables:
# mode: makefile
# End:

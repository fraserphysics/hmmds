SHELL=bash
# Rules.mk: This file can be included by a makefile anywhere as long
# as ROOT, HMMDS and BUILD are defined.  ROOT is the root of this
# project, HMMDS is where most code is, and BUILD is where derived
# results go.

LASER_DATA = $(BUILD)/derived_data/synthetic
LASER_CODE = $(HMMDS)/applications/laser
# LASER_CODE is this directory
RAW_DATA = $(ROOT)/raw_data
LP5DAT = $(RAW_DATA)/LP5.DAT

# Define POWELL250 as an abbreviation
POWELL250 = $(LASER_DATA)/ekf_powell250.parameters

$(LASER_DATA)/%.plot_data: $(LASER_CODE)/optimize_ekf.py $(LASER_DATA)/%.parameters
	python $< --method skip --plot_data $@ --laser_data $(LP5DAT) $(LASER_DATA)/$*.parameters

$(LASER_DATA)/pf_%.plot_data: $(LASER_CODE)/optimize_particle.py $(LASER_DATA)/%.parameters
	python $< --method skip --plot_data $@ --laser_data $(LP5DAT) $(LASER_DATA)/$*.parameters

$(LASER_DATA)/gui.plot_data: $(LASER_CODE)/optimize_ekf.py $(LASER_CODE)/explore.txt
	python $< --method skip --plot_data $@ --parameter_type GUI_out --laser_data $(LP5DAT) $(LASER_CODE)/explore.txt

$(LASER_DATA)/pf_ekf250.plot_data: $(LASER_CODE)/optimize_particle.py $(POWELL250)
	python $< --method skip --plot_data $@ --laser_data $(LP5DAT) $(POWELL250)

$(LASER_DATA)/pf_opt_noise.plot_data: $(LASER_CODE)/optimize_particle.py $(POWELL250) $(LASER_DATA)/pf_powell500.parameters
	python $< --method skip --plot_data $@ --laser_data $(LP5DAT) --override_parameters $(LASER_DATA)/pf_powell500.parameters $(POWELL250)

$(LASER_DATA)/pf_hand_noise.parameters: $(LASER_DATA)/pf_powell500.parameters
	sed -e "s/^state_noise.*/state_noise 0.2/" -e "s/^observation_noise.*/observation_noise 1.0/" < $< > $@

$(LASER_DATA)/pf_hand_noise.plot_data: $(LASER_CODE)/optimize_particle.py $(POWELL250) $(LASER_DATA)/pf_hand_noise.parameters
	python $< --method skip --plot_data $@ --laser_data $(LP5DAT) --override_parameters $(LASER_DATA)/pf_hand_noise.parameters $(POWELL250)

$(LASER_DATA)/%.particle_data: $(LASER_CODE)/optimize_particle.py $(LASER_DATA)/%.parameters
	python $< --method skip --n_particles 300 --laser_data $(LP5DAT) --plot_data $@ $(LASER_DATA)/$*.parameters

$(LASER_DATA)/pf_powell500.parameters: $(LASER_CODE)/optimize_particle.py $(POWELL250)
	python $< --length 175 --n_particles 500 --method Powell --laser_data $(LP5DAT) $(POWELL250) $@

# 5 hours -17.050910252989638 intermediate result, but final is -139.2018738220304?
$(LASER_DATA)/particle.parameters: $(LASER_CODE)/optimize_particle.py $(POWELL250)
	python $< --length 175 --n_particles 100 --method Powell --laser_data $(LP5DAT) $(POWELL250) $@

# Takes about 22 minutes
$(LASER_DATA)/ekf_powell2876.parameters: $(LASER_CODE)/optimize_ekf.py $(POWELL250)
	python $< --parameter_type parameter --laser_data $(LP5DAT) --length 2876 --method $(LASER_DATA)/Powell ekf_powell250.parameters $@

$(POWELL250): $(LASER_CODE)/optimize_ekf.py $(LASER_CODE)/explore.txt
	python $< --parameter_type GUI_out --laser_data $(LP5DAT) --length 250 --method Powell $(LASER_CODE)/explore.txt $@

# $(LASER_CODE)/explore.txt is under version control
$(LASER_CODE)/explore.txt: $(LASER_CODE)/explore.py
	echo Adjust sliders for period 5 orbit, match laser data, and press "write" button
	python explore.py
	mv explore.txt $@

# Pattern rule for LaserLP5 LaserLogLike LaserStates LaserForecast LaserHist
$(LASER_DATA)/Laser%: $(LASER_CODE)/figure_data.py $(POWELL250)
	python $< --parameters $(POWELL250) --laser_data $(LP5DAT) --Laser$* $@

# Local Variables:
# mode: makefile
# End:

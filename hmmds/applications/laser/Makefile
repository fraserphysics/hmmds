SHELL=bash

# f_max=-213.15338306475672
# success=False
# message=Desired error not necessarily achieved due to precision loss.
# iterations=2

#real	138m56.496s
#user	415m19.560s
#sys	0m16.864s

FIGURES = gui_plot.pdf ekf_powell250_plot.pdf ekf_powell2876_plot.pdf \
pf_ekf250_plot.pdf pf_opt_noise_plot.pdf pf_hand_noise_plot.pdf

laser_fit.pdf: laser_fit.tex $(FIGURES)
	pdflatex laser_fit.tex
	pdflatex laser_fit.tex

# Should depend on %.ekf_parameters ?
%.plot_data: optimize_ekf.py %.parameters
	python $< --method skip --plot_data $@ $*.parameters

pf_%.plot_data: optimize_particle.py %.parameters
	python $< --method skip --plot_data $@ $*.parameters

%_plot.pdf: plot.py %.plot_data
	python $^ $@

gui.plot_data: optimize_ekf.py explore.txt
	python $< --method skip --plot_data $@ --parameter_type GUI_out explore.txt

pf_ekf250.parameters: ekf_powell250.parameters
	ln -s $< $@

pf_opt_noise.plot_data: optimize_particle.py ekf_powell250.parameters pf_powell500.parameters
	python $< --method skip --plot_data $@ --override_parameters pf_powell500.parameters ekf_powell250.parameters

pf_hand_noise.parameters: pf_powell500.parameters
	sed -e "s/^state_noise.*/state_noise 0.2/" -e "s/^observation_noise.*/observation_noise 1.0/" < $< > $@

pf_hand_noise.plot_data: optimize_particle.py ekf_powell250.parameters pf_hand_noise.parameters
	python $< --method skip --plot_data $@ --override_parameters pf_hand_noise.parameters ekf_powell250.parameters

forecast_errors.pdf: cumulative.py pf_opt_noise.plot_data
	python $^ $@

%.particle_data: optimize_particle.py %.parameters
	python $< --method skip --n_particles 300 --plot_data $@ $*.parameters

pf_powell500.parameters: optimize_particle.py powell250.parameters
	python $< --length 175 --n_particles 500 --method Powell powell250.parameters $@

# 5 hours -17.050910252989638 intermediate result, but final is -139.2018738220304?
particle.parameters: optimize_particle.py powell250.parameters
	python $< --length 175 --n_particles 100 --method Powell powell250.parameters $@

# Takes about 22 minutes
powell2876.parameters: optimize_ekf.py powell250.parameters
	python $< --parameter_type parameter --laser_data LP5.DAT --length 2876 --method Powell powell250.parameters $@

powell250.parameters: optimize_ekf.py explore.txt
	python $< --parameter_type GUI_out --laser_data LP5.DAT --length 250 --method Powell explore.txt $@

explore.txt: explore.py
	echo Adjust sliders for period 5 orbit, match laser data, and press "write" button
	python explore.py

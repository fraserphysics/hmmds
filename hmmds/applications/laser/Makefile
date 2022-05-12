SHELL=bash

# f_max=-213.15338306475672
# success=False
# message=Desired error not necessarily achieved due to precision loss.
# iterations=2

#real	138m56.496s
#user	415m19.560s
#sys	0m16.864s

BFGS.parameters: optimize_particle.py powell250.parameters
	python $< --length 175 --n_particles 100 --method BFGS powell250.parameters $@

BFGS500.parameters: optimize_particle.py powell250.parameters
	python $< --length 175 --n_particles 500 --method BFGS powell250.parameters $@

particle.plot_data: optimize_particle.py particle.parameters
	python $< --method skip --n_particles 300 --plot_data $@ particle.parameters dummy
	rm dummy

powell500.parameters: optimize_particle.py powell250.parameters
	python $< --length 175 --n_particles 500 --method Powell powell250.parameters $@

# 5 hours -17.050910252989638 intermediate result, but final is -139.2018738220304?
particle.parameters: optimize_particle.py powell250.parameters
	python $< --length 175 --n_particles 100 --method Powell powell250.parameters $@

test_particle: particle.py
	time python particle.py --n_times 2876 powell2876.parameters $@
particle_result_plot.pdf: plot.py particle_result
	python $^ $@
#time real	17m43.470s
particle_result: particle.py powell250.parameters
	python $^ $@

# Takes about 22 minutes
powell2876.parameters: optimize_ekf.py powell250.parameters
	python $< --parameter_type parameter --laser_data LP5.DAT --length 2876 --method Powell powell250.parameters $@

powell250.parameters: optimize_ekf.py explore.txt
	python $< --parameter_type GUI_out --laser_data LP5.DAT --length 250 --method Powell explore.txt $@

%.particle_data: optimize_particle.py %.parameters
	python $< --method skip --n_particles 300 --plot_data $@ $*.parameters dummy
	rm dummy
%_plot.pdf: plot.py %.particle_data
	python $^ $@

%.plot_data: optimize_ekf.py %.parameters
	python $< --method skip --plot_data $@ $*.parameters dummy
	rm dummy
%_plot.pdf: plot.py %.plot_data
	python $^ $@

explore.txt: explore.py
	echo Adjust sliders for period 5 orbit, match laser data, and press "write" button
	python explore.py

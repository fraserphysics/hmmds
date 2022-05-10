SHELL=bash
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
%.plot_data: optimize_ekf.py %.parameters
	python $< --method skip --plot_data $@ $*.parameters dummy
	rm dummy
%_plot.pdf: plot.py %.plot_data
	python $^ $@

explore.txt: explore.py
	echo Adjust sliders for period 5 orbit, match laser data, and press "write" button
	python explore.py

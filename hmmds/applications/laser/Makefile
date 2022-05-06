
# Default target
powell2876_plot.pdf:

# Takes about 22 minutes
powell2876.parameters: optimize.py powell250.parameters
	python $< --parameter_type parameter --laser_data LP5.DAT --length 2876 --method Powell powell250.parameters $@

powell250.parameters: optimize.py explore.txt
	python $< --parameter_type GUI_out --laser_data LP5.DAT --length 250 --method Powell explore.txt $@
%.plot_data: optimize.py %.parameters
	python $< --method skip --plot_data $@ $*.parameters dummy
	rm dummy
%_plot.pdf: plot.py %.plot_data
	python plot.py --data $*.plot_data --fig_path $@

explore.txt: explore.py
	echo Adjust sliders for period 5 orbit, match laser data, and press "write" button
	python explore.py
particle.pdf: plot.py test_particle
	python $< --data test_particle --fig_path $@

ekf.pdf: plot.py test_ekf
	python $< --data test_ekf --fig_path $@

test_ekf: ekf.py optimize.py LP5.DAT
	python $< --LaserData LP5.DAT --result $@

test_particle: particle.py
	python $< --LaserData LP5.DAT --result $@

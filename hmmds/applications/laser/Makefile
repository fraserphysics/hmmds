ekf.pdf: plot.py test_ekf
	python $< --data test_ekf --fig_path $@

test_ekf: ekf.py optimize.py LP5.DAT
	python $< --LaserData LP5.DAT --result $@

test_particle: particle.py
	python $< --LaserData LP5.DAT --result $@

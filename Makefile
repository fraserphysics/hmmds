N_TRAIN = 50

figs/pass1.pdf: plotscripts/apnea/pass1.py derived_data/apnea/pass1_report.pickle
	python $^ $@

derived_data/apnea/pass1_report.pickle:
	cd hmmds/applications/apnea; make pass1_report.pickle

figs/Statesintro.pdf: plotscripts/stateplot.py derived_data/synthetic/states
	python  $< --data_dir derived_data/synthetic --base_name state --fig_path $@

derived_data/synthetic/states: hmmds/synthetic/StatePic.py derived_data/synthetic/m12s.4y
	python $<  derived_data/synthetic lorenz.4 lorenz.xyz m12s.4y

derived_data/synthetic/m12s.4y : hmmds/synthetic/MakeModel.py derived_data/synthetic/lorenz.xyz
	python hmmds/synthetic/MakeModel.py ${N_TRAIN} derived_data/synthetic lorenz.4 m12s.4y

derived_data/synthetic/lorenz.xyz: hmmds/synthetic/lorenz.py
	python $< --n_samples 20000 --levels 4 --quantfile derived_data/synthetic/lorenz.4 --xyzfile $@


## yapf                           : Force google format on all python code
.PHONY : yapf
yapf :
	yapf -i --recursive --style "google" hmmds

## help                           : Print comments on targets from makefile
.PHONY : help
help : Makefile
	@sed -n 's/^## / /p' $<

# Local Variables:
# mode: makefile
# End:

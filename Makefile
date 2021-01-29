N_TRAIN = 10

figs/Statesintro.pdf: hmmds/plotscripts/stateplot.py derived_data/synthetic/states
	python  hmmds/plotscripts/stateplot.py --data_dir derived_data/synthetic --base_name state --fig_path $@

derived_data/synthetic/states: hmmds/synthetic/StatePic.py derived_data/synthetic/m12s.4y
	python $<  derived_data/synthetic lorenz.4 lorenz.xyz m12s.4y

derived_data/synthetic/m12s.4y : hmmds/synthetic/MakeModel.py derived_data/synthetic/lorenz.xyz
	python hmmds/synthetic/MakeModel.py ${N_TRAIN} derived_data/synthetic lorenz.4 m12s.4y

derived_data/synthetic/lorenz.xyz: hmmds/synthetic/lorenz.py
	python $< --L 20000 --levels 4 --quantfile derived_data/synthetic/lorenz.4 --xyzfile $@

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

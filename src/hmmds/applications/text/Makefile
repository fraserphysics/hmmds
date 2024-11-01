DATA_DIR = ../../../../raw_data/
foo.tex: po_speech.py $(DATA_DIR)/menken.txt
	file  $(DATA_DIR)/menken.txt
	python $^ --n_iterations 100 --random_seed 1 $@

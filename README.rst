
Hidden Markoc Models and Dynamical Systems
==========================================

HMMDS3 provides python3 code that implements the following algorithms
for hidden Markov models:

Forward: Recursive estimation of state probabilities at each time t,
         given observation likelihoods for times 1 to t

Backward: Combined with Forward, provides estimates of state
          probabilities at each time given _all_ of the observation
          likelihoods

Train: Implements Baum Welch algorithm which finds a local maximum of
       likelihood of model parameters

Decode: Implements Viterbi algorithm for finding the most probable
        state sequence

Implementations of the above algrithms are independent of the
observation model.  HMMDS3 enables users to implement any observation
model by writing code for a class that provides methods for
calculating the likelihood of an observation given a state and for
reestimating model parameters given observations and state
likelihoods.

HMMDS3 includes implementations of the following observation models:

Discrete: Integers in a finite range

Gauss: Floats with state dependent mean and variance

Class_Y: Observations that can include classification data

I (Andy Fraser) started this project on 2012-10-31.  I will rewrite
the code for my book "Hidden Markov Models and Dynamical Systems"
using the following tools: python3, numpy, scipy, sphinx, doctest, qt4

and not using: gnuplot, swig, c

I will sacrifice the appearance of plots to simplicity of the code.  I
hope that the scipy sparse matrix package will let me have simple fast
code.  I will choose simplicity over speed within reason.

My starting point is the project "webhmmds" in svn at
http://fraserphysics.com/webhmmds.  After I get the hmm code with most
output model options running and documented/tested with
sphinx/doctest, I will put the project on google_code under git and
GPL3.  I will use a single brief LaTeX document instead of the text
that SIAM owns to collect the figures.

https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
http://www.python.org/dev/peps/pep-0008/

To do:

1. Test new class decode

2. Run apnea application to completion

3. Problem with import hmm.C using __init__.py package stuff

4. Get book to build

5. From single source (HMMDS3) make two distribution tarballs: HMMpy
   and HMM_DS

6. Clean up EKF and put it in code/hmm.  Reduce it to Forward(),
   Backward(), and Smooth().  Write test functions.

7. Do the To Do list on page v of software.pdf

-1. Get 2 axes rather than box for matplotlib plots and kill boxes on legends

-2. Remember bug in argparse that makes empty file at default argument

-3. Get __module__ or something like it to include "hmm/" so that
    pickle works without including  "/home/andy/..../hmm/" in
    PYTHONPATH

-4. Get sphinx to parse cython

-5. Study alternative build systems

To run scripts from the command line rather than scons, one must
modify the PYTHONPATH environment variable.  The following lines work
for my configuration:

    export PYTHONPATH=/home/andy/projects/hmmds3/code/
    export PYTHONPATH=/home/andy/projects/hmmds3/code/hmm/:$PYTHONPATH

While I did most of the work on this code at FraserPhysics, a I did a
fraction at Los Alamos National Laboratory (LANL).  The "Los Alamos
Computer Code" for this work is LA-CC-13-008.

This file is part of hmmds3.
Copyright 2013 Andrew M. Fraser.

You can redistribute and/or modify hmmds3 under the terms of the GNU
General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later
version.  See the file "License" in the root directory of the
hmmds3 distribution.

--------------------------------------------------------------------
Run:

conda create -n hmmds python=3.5 --file pip_req.txt
source activate hmmds
python setup.py develop

# Local Variables:
# mode: rst
# End:


Hidden Markov Models and Dynamical Systems
==========================================

HMMDS provides python3 code that implements the following algorithms
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
observation model.  HMMDS enables users to implement any observation
model by writing code for a class that provides methods for
calculating the likelihood of an observation given a state and for
reestimating model parameters given observations and state
likelihoods.

HMMDS includes implementations of the following observation models:

Discrete: Integers in a finite range

Gauss: Floats with state dependent mean and variance

Class_Y: Observations that can include classification data

I (Andy Fraser) restarted this project on 2017-8-4.  I will rewrite
the code for my book "Hidden Markov Models and Dynamical Systems"
using the following tools: python3, make, numpy, scipy, sphinx, pytest, qt5

and not using: gnuplot, swig, c, scons

I will sacrifice the appearance of plots to simplicity of the code.  I
hope that the scipy sparse matrix package will let me have simple fast
code.  I will choose simplicity over speed within reason.

For development, I am using the anaconda package manager from
Continuum Analytics, see: https://www.continuum.io/downloads

My starting point is the project "webhmmds" in svn at
http://fraserphysics.com/webhmmds.  I will use a single brief LaTeX
document instead of the text that SIAM owns to collect all of the
figures.

While I did most of the work on this code at FraserPhysics, a I did a
fraction at Los Alamos National Laboratory (LANL).  The "Los Alamos
Computer Code" for this work is LA-CC-13-008.

This file is part of hmmds.
Copyright 2013 and 2017 Andrew M. Fraser.

You can redistribute and/or modify hmmds under the terms of the GNU
General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later
version.  See the file "License" in the root directory of the
hmmds distribution.

--------------------------------------------------------------------
Run:

conda create -n hmmds python=3.5 --file pip_req.txt
source activate hmmds
python setup.py develop

# Local Variables:
# mode: rst
# End:

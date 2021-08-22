Although the primary goal of this project is to create an improved
version of my book Hidden Markov Models and Dynamical Systems, it
includes other stuff, eg my SIAM DS-21 presentation.  The code here
relies on the package hmm which is available as hmm4ds from PyPi.

The directories here are:

TeX:  LaTeX source

    I want a separate sub-directory with a makefile for each
    independent document

derived_data:  The name describes it.  Sub-directories for different kinds

figs: Sub-directories contain derived figure files.  There shouldn't
    be any figure files in the top level of figs/.

hmmds: An importable python project that contains most of the code
    except calls to matplotlib.

plot_scripts: Xfigs and python that calls matplotlib

raw_data: Data fetched from elsewhere

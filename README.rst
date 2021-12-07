HMMDS: Hidden Markov Models and Dynamical Systems
=================================================

Although the primary goal of this project is to create an improved
version of my book `Hidden Markov Models and Dynamical Systems
<https://epubs.siam.org/doi/book/10.1137/1.9780898717747?mobileUi=0>`_,
it includes other stuff, eg my SIAM DS-21 presentation.  The code here
relies on a package available from PyPi as hmm4ds with source at
gitlab called `hmm <https://gitlab.com/fraserphysics1/hmm>`_.

The files and directories here are:

* *Makefile*: Controls building the book via gnu-make.  Results from
  invoking *make* appear in the directory *build/*.
* *README.rst*: This file.
* *TeX/*: LaTeX source.  There is a separate sub-directory with a
  makefile for each independent document.
* *hmmds/*: Code for calculations and data manipulation.
* *plotscripts/*: Code for making figures.  All code that uses
  matplotlib is segregated here.
* *pylintrc*: Defines coding style/standards for the project.

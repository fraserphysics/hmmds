HMMDS: Hidden Markov Models and Dynamical Systems
=================================================

The primary goal of this project is to create an improved version of
my book `Hidden Markov Models and Dynamical Systems
<https://epubs.siam.org/doi/book/10.1137/1.9780898717747?mobileUi=0>`_.
This is work in progress.  I can build a draft of the book by
issuing ``make book`` in the root directory.  Several other targets in
the Makefile in the root directory build smaller documents, eg,
``make ddays25`` builds the poster that I presented at Dynamics Days US 2025
in Denver.  Try ``make help`` to see a list of most of those targets.

I work in a `Nix <https://nixos.org/>`_ environment, and as of January
2025 the code here is not portable to other environments.  The code
here relies on the hmm package available at `hmm
<https://gitlab.com/fraserphysics1/hmm>`_.

The files and directories here are:

* *Makefile*: Controls building the book via gnu-make.  Results from
  invoking *make* appear in the directory *build/*.
* *README.rst*: This file.
* *src/TeX/*: LaTeX source.  There is a separate sub-directory with a
  Rules.mk for each independent document.
* *src/hmmds/*: Code for calculations and data manipulation.
* *src/plotscripts/*: Code for making figures.  Code that uses
  matplotlib is segregated here.
* *pylintrc*: Defines coding style/standards for the project.

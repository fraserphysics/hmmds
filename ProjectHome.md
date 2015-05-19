# About #
Derived from the code for Fraser's book "Hidden Markov Models and Dynamical Systems", hmmds provides an HMM class that supports customized output/observation models.  We include code and data for all examples and figures in the book.  We have written the base classes in python3 for clarity and used cython and scipy sparse matrices for efficiency and speed.

### Download ###
To fetch the source execute
```
git clone https://code.google.com/p/hmmds/
```
See http://code.google.com/p/hmmds/source/checkout for details.

### Use ###
The code fetched by the above command builds `TeX/software.pdf` using a command like
```
scons -j 6 TeX/software.pdf
```
which builds the document on my (Andy Fraser) 8 cpu system in about 150 minutes.  The document _software.pdf_ describes how each figure in _the book_ is built.

I intend to divide the code into two parts, one that provides general purpose hmm tools, and another that builds _software.pdf_.  For now, you can find most of the general purpose hmm code in the directory `code/hmm`.

### System Requirements ###
I developed the code on 64 bit Linux platforms running _debian testing_.  Building _software.pdf_ requires about 3GB of memory, but you can use the software for smaller problems and get by with less memory.

I have not yet worked on making the code run on other platforms.

# Time Line #



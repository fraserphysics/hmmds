'''Master file for scons, a software build tool.

Would need the following to work with scipy in same process:

    >export SCONS_HORRIBLE_REGRESSION_TEST_HACK=yes

But, I run all commands in sub-processes because most of the code is
for python3 and scons does not run under python3.

SCons User Guide at
http://www.scons.org/doc/production/HTML/scons-user/index.html

Wiki at
www.scons.org/wiki

Documentation of LaTeX scanner class at
http://www.scons.org/doc/HTML/scons-api/SCons.Scanner.LaTeX.LaTeX-class.html

http://www.scons.org/wiki/LatexSupport

'''
Copyright='''
    Copyright 2013 Andrew M. Fraser and Los Alamos National Laboratory
    
    This file is part of hmmds3.

    Hmmds3 is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Hmmds3 is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See the file gpl.txt in the root directory of the hmmds3
    distribution or see <http://www.gnu.org/licenses/>.
'''
def build_pdf_t(target, source, env):
    ''' Written for the fig2pdf Builder, this function runs fig2dev
    twice on an xfig source.  Prolly belongs in plotscript dir.
    "target" is two Nodes [*.pdf, *.pdf_t]
    "source" is single Node [*.fig]
    '''
    import subprocess
    from os.path import basename
    x_fig = str(source[0])
    x_pdf = str(target[0])
    x_pdf_t = str(target[1])
    for L in (['fig2dev', '-L', 'pdftex', x_fig, x_pdf],
              ['fig2dev', '-L', 'pdftex_t', '-p', basename(x_pdf),
               x_fig, x_pdf_t]):
        try:
            r = subprocess.call(L)
            if r != 0:
                print('''
Error: In build_pdf_t(), subprocess.call(%s)
    returned %d
'''%(r, L))
                return -1
        except Exception as e:
            print('''
Error: In build_pdf_t(), subprocess.call(%s)
     raised %s
'''%(L, e))
            return -1
    return None

'''fig2pdf is a SCons Builder for making "%.pdf" and "%.pdf_t" from
"%.fig".  The arguments of the emitter function, "target" and
"source", are lists of SCons Nodes.'''
fig2pdf = Builder(
    action=build_pdf_t, src_suffix='.fig', suffix='.pdf',
    emitter=lambda target, source, env: ([target[0], str(target[0])+'_t'],
                                         source))

from os.path import join
C  = lambda file: join(GetLaunchDir(),'code/',file)
CH  = lambda file: join(GetLaunchDir(),'code/hmm/',file)
CAS = lambda file: join(GetLaunchDir(),'code/applications/synthetic/', file)
CAL = lambda file: join(GetLaunchDir(),'code/applications/laser/', file)
CAA = lambda file: join(GetLaunchDir(),'code/applications/apnea/', file)
CAO = lambda file: join(GetLaunchDir(),'code/applications/other/', file)
CPS = lambda file: join(GetLaunchDir(),'code/plotscripts/', file)
CXF = lambda file: join(GetLaunchDir(),'code/xfigs/', file)
DD = lambda file: join(GetLaunchDir(),'derived_data/', file)
DDS = lambda file: join(GetLaunchDir(),'derived_data/synthetic/', file)
DDA = lambda file: join(GetLaunchDir(),'derived_data/apnea/', file)
DDL = lambda file: join(GetLaunchDir(),'derived_data/laser/', file)
RD = lambda file: join(GetLaunchDir(),'raw_data/', file)
RDA = lambda file: join(GetLaunchDir(),'raw_data/apnea/', file)
FIG = lambda file: join(GetLaunchDir(),'figs', file)

#  CH('') required by pickle in
#  code/applications/synthetic/MakeModel.py hmm.C.HMM instance has
#  C.HMM as value of self.__class__ but hmm.base.HMM instatnce has
#  hmm.base.HMM as value of self.__class__ ?
ENV = {'PYTHONPATH':'%s:%s:%s:%s'%(CAS(''), C(''), CH(''), CAA(''))}

def KEY(target):
    '''Constuct a key from the first target.  The key gets used to find
    the arguments in the dict env.args.  I use the last two elements
    of the path.  While one element is ambiguous for pairs like
    derived_data/apnea/respiration/a01 and
    derived_data/apnea/r_times/a01, two elements are sufficient.

    '''
    import os.path
    a = os.path.basename(str(target[0]))
    d = os.path.dirname(str(target[0]))
    return (a, os.path.basename(d))
def BUILD(target, source, env):
    '''SConscripts use BUILD and KEY to have python invoked on specific
    scripts with specific arguments.  Each environment has a
    dictionary (env.args) of arguments for BUILD to use.  The keys are
    the first targets that the command will build.  KEY() extracts the
    basename as a simple string to get around scons' re-typing.  The
    script to run must be source[0].

    '''
    from subprocess import call
    print('call %s'%source[0])
    call(('python3', str(source[0])) + tuple(env.args[KEY(target)]), env=ENV)
def BUILD2(target, source, env):
    '''Need python2 for plotting because matplotlib for python3 doesn't
    exist.  Also I import the current environment because matplotlib
    needs something from the shell environment.

    '''
    from subprocess import call
    import os
    d = {}
    d.update(os.environ)
    d.update(ENV)
    print('call %s'%source[0]) # Name "env" used twice.  Different and OK.
    call(('python2.7', str(source[0])) + env.args[KEY(target)], env=d)

SConscript(CAS('SConscript'), exports='CH RD DDS KEY BUILD') # Apps/Synthetic
SConscript(CAA('SConscript'), exports='DDA RDA KEY BUILD')   # Apps/Apnea
SConscript(CAO('SConscript'), exports='DD RD CH KEY BUILD')  # Apps/Other
# Following line is plot scripts
SConscript(CPS('SConscript'), exports='DDL DDA DDS RDA RD FIG KEY BUILD2')
SConscript(CXF('SConscript'), exports='fig2pdf FIG') # xfigs

# The remaining code fragments are so small that I have not put them
# in SConscript files.
swe=Environment()
software = swe.PDF('TeX/software.tex')
swe.PDF('TeX/ii.tex')
swe['TEXINPUTS'] = ['figs','TeX', 'derived_data']
# Added dependencies that scan of software.tex misses
Depends(software, [
    'figs/Markov_mm.pdf_t',
    'derived_data/po_speech',
    'figs/pass1.pdf'])

env=Environment()
env.args = {}
env.Command(
    CH('C.cpython-32mu.so'),
    (CH('Scalar.py'), CH('C.pyx')),
    'cd %s; python3 setup.py build_ext --inplace'%CH('')
    )
env.Command(
    CAS('lor_C.cpython-32mu.so'),
    (CAS('lor_C.pyx'),),
    'cd %s; python3 setup.py build_ext --inplace'%CAS('')
    )
target = tuple(DDL(x) for x in ('LaserLP5', 'LaserForecast', 'LaserHist',
                           'LaserLogLike', 'LaserStates'))
source = tuple((CAL('Laser_data.py'), RD('LP5.DAT')))
env.args[KEY(target)] = (RD('LP5.DAT'), DDL(''))
#env.Command(target, source, BUILD)

#---------------
# Local Variables:
# eval: (python-mode)
# End:

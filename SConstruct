'''Master file for scons, a software build tool.

Would need the following to work with scipy in same process:

    >export SCONS_HORRIBLE_REGRESSION_TEST_HACK=yes

But, I run all commands in sub-processes because most of the code is
for python3 and scons doesn't run under python3.

SCons User Guide at
http://www.scons.org/doc/production/HTML/scons-user/index.html

Wiki at
www.scons.org/wiki

Documentation of LaTeX scanner class at
http://www.scons.org/doc/HTML/scons-api/SCons.Scanner.LaTeX.LaTeX-class.html

http://www.scons.org/wiki/LatexSupport

'''

def build_pdf_t(target, source, env):
    ''' Written for the fig2pdf Builder, this function runs fig2dev
    twice on an x-fig source.  Prolly belongs in plotscript dir.
    "target" is two Nodes [*.pdf, *.pdf_t]
    "source" is single Node [*.fig]
    '''
    import subprocess
    x_fig = str(source[0])
    x_pdf = str(target[0])
    x_pdf_t = str(target[1])
    subprocess.call(['fig2dev', '-L', 'pdftex', x_fig, x_pdf])
    subprocess.call(['fig2dev', '-L', 'pdftex_t', '-p', x_pdf, x_fig, x_pdf_t])
    return None

'''fig2pdf is a SCons Builder for making "%.pdf" and "%.pdf_t" from
"%.fig".  The arguments of the emitter function, "target" and
"source", are lists of SCons Nodes.'''
fig2pdf = Builder(
    action=build_pdf_t, src_suffix='.fig', suffix='.pdf',
    emitter=lambda target, source, env: ([target[0], str(target[0])+'_t'],
                                         source))

from os.path import join
CH  = lambda file: join(GetLaunchDir(),'code/hmm/',file)
CAS = lambda file: join(GetLaunchDir(),'code/applications/synthetic/', file)
CAA = lambda file: join(GetLaunchDir(),'code/applications/apnea/', file)
CPS = lambda file: join(GetLaunchDir(),'code/plotscripts/', file)
DDS = lambda file: join(GetLaunchDir(),'derived_data/synthetic/', file)
DDA = lambda file: join(GetLaunchDir(),'derived_data/apnea/', file)
RDA = lambda file: join(GetLaunchDir(),'raw_data/apnea/', file)
FIG = lambda file: join(GetLaunchDir(),'figs', file)

ENV = {'PYTHONPATH':'%s:%s:%s'%(CAS(''), CH(''), CAA(''))}

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

SConscript(CAS('SConscript'), exports='CH DDS KEY BUILD')           # Synthetic
SConscript(CAA('SConscript'), exports='DDA RDA KEY BUILD')          # Apnea
SConscript(CPS('SConscript'), exports='DDA DDS RDA FIG KEY BUILD2') # plots

# The remaining code fragments are so small that I have not put them
# in SConscript files.
swe=Environment()
swe.PDF('TeX/software.tex')
swe['TEXINPUTS'] = ['figs','TeX']

env=Environment()
env.Command(
    CH('C.cpython-32mu.so'),
    (CH('Scalar.py'), CH('C.pyx')),
    'cd %s; python3 setup.py build_ext --inplace'%CH('')
    )

#---------------
# Local Variables:
# eval: (python-mode)
# End:

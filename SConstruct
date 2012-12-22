''' Master file for scons, a software build tool.

Need the following to work with scipy
>export SCONS_HORRIBLE_REGRESSION_TEST_HACK=yes

SCons User Guide at
http://www.scons.org/doc/production/HTML/scons-user/index.html

Wiki at
www.scons.org/wiki

Documentation of LaTeX scanner class at
http://www.scons.org/doc/HTML/scons-api/SCons.Scanner.LaTeX.LaTeX-class.html

http://www.scons.org/wiki/LatexSupport

Goal: SConscript files in sub-directories should have functions for
  paths to data in other directories passed.  The character "/" should
  not appear in an SConscript file.
'''

from os.path import join
def build_pdf_t(target, source, env):
    ''' Written for the fig2pdf Builder, this function runs fig2dev
    twice on an x-fig source.
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

CH  = lambda file: join(GetLaunchDir(),'code/hmm/',file)
CAS = lambda file: join(GetLaunchDir(),'code/applications/synthetic/', file)
CAA = lambda file: join(GetLaunchDir(),'code/applications/apnea/', file)
CPS = lambda file: join(GetLaunchDir(),'code/plotscripts/', file)
DDS = lambda file: join(GetLaunchDir(),'derived_data/synthetic/', file)
DDA = lambda file: join(GetLaunchDir(),'derived_data/apnea/', file)
RDA = lambda file: join(GetLaunchDir(),'raw_data/apnea/', file)
FIG = lambda file: join(GetLaunchDir(),'figs', file)

PYTHON = 'env PYTHONPATH=code/applications/synthetic:code/hmm python3 '
PYTHON = 'env PYTHONPATH=%s:%s python3 '%(CAS(''),CH(''))
#Need python2 for plotting because matplotlib for python3 doesn't exist
PYTHON2 = 'env PYTHONPATH=%s python2.7 '%(CAA(''))

SConscript(CAS('SConscript'), exports='PYTHON CH DDS')
SConscript(CAA('SConscript'), exports='PYTHON DDA RDA')
SConscript(CPS('SConscript'), exports='PYTHON2 DDA DDS RDA FIG')

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

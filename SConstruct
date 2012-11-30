'''SConstruct: Derived from eos project

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
    subprocess.call(['fig2dev','-L','pdftex',x_fig,x_pdf])
    subprocess.call(['fig2dev', '-L', 'pdftex_t', '-p', x_pdf, x_fig, x_pdf_t])
    return None

'''fig2pdf is a SCons Builder for making "%.pdf" and "%.pdf_t" from
"%.fig".  The arguments of the emitter function, "target" and
"source", are lists of SCons Nodes.'''
fig2pdf = Builder(
    action=build_pdf_t, src_suffix='.fig', suffix='.pdf',
    emitter=lambda target,source,env:([target[0],str(target[0])+'_t'],source))

PYTHON = 'env '+\
    ' PYTHONPATH=code/applications/synthetic:code/hmm '+\
    ' SCONS_HORRIBLE_REGRESSION_TEST_HACK=no '+\
    ' python3 '
#Need python2 for plotting because matplotlib for python3 doesn't exist
PYTHON2 = ('env '+\
    ' PYTHONPATH=%s '+\
    ' SCONS_HORRIBLE_REGRESSION_TEST_HACK=no '+\
    ' python2.7 ')%('code/applications/apnea',)
CH  = lambda file: join('code/hmm/',file)
CAS = lambda file: join('code/applications/synthetic/', file)
CAA = lambda file: join('code/applications/apnea/', file)
CPS = lambda file: join('code/plotscripts/', file)
DDS = lambda file: join('derived_data/synthetic/', file)
DDA = lambda file: join('derived_data/apnea/', file)
RDA = lambda file: join('raw_data/apnea/', file)

SConscript('SConscript', exports='PYTHON PYTHON2 CH CAS CPS DDS')
SConscript('code/applications/apnea/SConscript', exports='PYTHON DDA RDA')
SConscript('code/plotscripts/SConscript', exports='PYTHON2 DDA RDA')

#---------------
# Local Variables:
# eval: (python-mode)
# End:

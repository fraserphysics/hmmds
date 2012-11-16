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
'''

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
PYTHON2 = 'env '+\
    ' SCONS_HORRIBLE_REGRESSION_TEST_HACK=no '+\
    ' python2.7 '
CH  = lambda file: 'code/hmm/'+file
CAS = lambda file: 'code/applications/synthetic/'+file
CPS = lambda file: 'code/plotscripts/'+file
DDS = lambda file: 'derived_data/synthetic/'+file
STATEDATA = [DDS('states')]+[DDS('state%s'%x) for x in range(12)]
swe=Environment()
swe.PDF('TeX/software.tex')
swe['TEXINPUTS'] = ['figs','TeX']
swe.Command(
    [DDS('lorenz.xyz'),DDS('lorenz.4')],CAS('lorenz.py'),
    PYTHON+'%s --L=20000 --levels=4 --quantfile=%s --xyzfile=%s'%(
        CAS('lorenz.py'),DDS('lorenz.4'),DDS('lorenz.xyz'))
    )
swe.Command(
    CH('C.cpython-32mu.so'),
    [CH('Scalar.py'), CH('C.pyx')],
    'cd code/hmm; python3 setup.py build_ext --inplace'
    )
swe.Command(
    DDS('m12s.4y'),
    [CH('C.cpython-32mu.so'), CH('Scalar.py'), CAS('MakeModel.py'),
     DDS('lorenz.4')],
    PYTHON+CAS('MakeModel.py')+' 500 derived_data/synthetic lorenz.4 m12s.4y'
    )
swe.Command(
    STATEDATA,
    [DDS('m12s.4y')]+[DDS('lorenz.4')]+[CPS('StatePic.py')],
    PYTHON+CPS('StatePic.py')+
      ' derived_data/synthetic lorenz.4 lorenz.xyz m12s.4y'
    )
swe.Command(
    'figs/Statesintro.pdf',                                # target
    [CPS('stateplot.py')]+STATEDATA,                       # sources
    PYTHON2+CPS('stateplot.py')+                           # command
      ' derived_data/synthetic state figs/Statesintro.pdf'
    )
#---------------
# Local Variables:
# eval: (python-mode)
# End:

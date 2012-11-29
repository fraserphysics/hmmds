'''SConscript
'''
Import('PYTHON PYTHON2 CH CAS CPS DDS')
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

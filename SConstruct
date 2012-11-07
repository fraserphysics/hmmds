'''SConstruct: Derived from eos project

SCons User Guide at
http://www.scons.org/doc/production/HTML/scons-user/index.html

Wiki at
www.scons.org/wiki

Documentation of LaTeX scanner class at
http://www.scons.org/doc/HTML/scons-api/SCons.Scanner.LaTeX.LaTeX-class.html

http://www.scons.org/wiki/LatexSupport
'''
import numpy, scipy

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

swe=Environment()
swe.PDF('TeX/software.tex')

#---------------
# Local Variables:
# eval: (python-mode)
# End:
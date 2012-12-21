""" PFsurvey.py: Creates Fig. 6.10 of the book

python PFsurvey.py PFdata PFplot
"""
Debug = True
import numpy as np
def read_data(data_file):
    x_dict = {}
    for line in open(data_file, 'r'):
        x, y, z = (float(p) for p in line.split())
        if not x_dict.has_key(x):
            x_dict[x] = {}
        x_dict[x][y] = z
    xs = list(x_dict.keys())
    xs.sort()
    xs = np.array(xs)
    ys = x_dict[xs[0]].keys()
    ys.sort()
    ys = np.array(ys)
    n_x = len(xs)
    n_y = len(ys)
    zs = np.empty((n_y, n_x))
    for i in range(n_x):
        for j in range(n_y):
            zs[j,i] = x_dict[xs[i]][ys[j]] # FixMe: Why is this right?
    return xs, ys, zs

import sys
def main(argv=None):
    '''Call with arguments: data, fig_file

    data is a text file with lines like the following:
        0.500  0.500  0.5024
        0.500  1.500  0.9196
        0.500  2.500  0.9280
        0.500  3.500  0.9272
        0.500  4.500  0.9274
        1.500  0.500  0.5480

    fig_file is a path where this script writes the result.

    '''
    if sys.version_info < (3,0):
        import matplotlib as mpl
        if not Debug:
            mpl.use('PDF')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # for  "projection='3d'".
    else:
       print('%s needs matplotlib.  However, no matplotlib for python %s'%(
           sys.argv[0],sys.version_info,))
       return -1

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    data_file, figure_file = argv
    pass1_report, fig_name = argv
    data = read_data(pass1_report)
    params = {'axes.labelsize': 12,
                   'text.fontsize': 10,
                   'legend.fontsize': 10,
                   'xtick.labelsize': 11,
                   'ytick.labelsize': 11,
                   'text.usetex': True,}
    if Debug:
        params['text.usetex'] = False
    mpl.rcParams.update(params)

    fig = plt.figure(figsize=(8, 4))
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(1, 1, 1, projection='3d', elev=30, azim=-115)
    ax.set_xlabel('Power')
    #ax.set_xlim(-2, 9)
    ax.set_ylabel('Fudge')
    #ax.set_ylim(1.4, 3.2)
    xs, ys, zs = read_data(data_file)
    X, Y = np.meshgrid(xs, ys)
    surf = ax.plot_surface(X, Y, zs, rstride=1, cstride=1,
                   cmap=mpl.cm.jet, linewidth=0, antialiased=False)
    #ax.set_zlim(.85, 0.9)
    if Debug:
        plt.show()
    else:
        fig.savefig(fig_name)
    return 0

if __name__ == "__main__":
    args = ['pf_M', 'pf_M.pdf']
    sys.exit(main(args))
# Local Variables:
# mode: python
# End:

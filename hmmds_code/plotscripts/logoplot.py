"""The script that makes the logo for google code.  Since this uses
matplotlib, it **cannot run under python3**.

"""

import sys
def main(argv=None):
    '''Call with arguments: data_dir, base_name, fig_name

    data_dir is the directory that has the state files

    base_name When it is "state" the data files are "state0", "state1"
    ,..., "state11".

    fig_name, eg, figs/Statesintro.pdf.  Where the figure gets written

    '''

    if sys.version_info < (3,0):
        import matplotlib as mpl
        #mpl.use('PDF')
        import matplotlib.pyplot as plt
    else:
       print('%s needs matplotlib.  However, no matplotlib for python %s'%(
           sys.argv[0],sys.version_info,))
       return -1

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    data_dir, base_name, fig_name = argv
    plotcolor = [
        [1,0,0],       # Red
        [0,1,0],       # Green
        [0,0,1],       # Blue
        [0,1,1],       # Cyan
        [1,0,1],       # Magenta
        [.95,.95,0],  # Yellow
        [0,0,0]        # Black
        ]
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(1,1,1)
    for b in range(0,12):
        name = '%s/%s%d'%(data_dir,base_name,b)
        xlist = []
        ylist = []
        zlist = []
        for line in open(name, 'r').readlines():#Read the data again
            x,y,z = [float(w) for w in line.split()]
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(xlist,zlist,color=plotcolor[b%7],marker=',',markersize=2,
                linestyle='None')
    ax.set_xlim(-20,20)
    ax.set_ylim(0,50)
    fig.savefig(fig_name)
    return 0

if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:

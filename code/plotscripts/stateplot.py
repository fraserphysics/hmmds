"""The script that makes the cover figure.  Invocation

python stateplot.py derived_data/synthetic state figs/Statesintro.pdf
"""
import string, sys, matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import pylab

def main(argv=None):

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
    pylab.rc('figure', figsize=(5,5))
    num = 0
    skiplist = [1,2,5,6]
    #The first loop is to graph each individual set of points, the second
    #is to get all of them at once.
    for b in range(0,12): #The last file is state11.
        xlist = []
        ylist = []
        zlist = []
        name = data_dir + '/'+base_name + str(b)
        f = open(name, 'r') #Read the data file
        lines = f.readlines() #count out how many lines there are
        for line in lines:             #At each line, split the line into
                                       #the X, Y, and Z coordinates,
            words = string.split(line) #then add each of them to their own list.
            x = float(words[0])
            y = float(words[1])
            z = float(words[2])
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
        num += 1
        while num in skiplist: #This is to make space for putting in the
                               #figure with all the assembled pieces.
            num += 1

        pylab.subplot(4,4,num) #There are different subplots between here
                               #and the next loop. One is 4x4 with smaller
                               #pieces, the other is a 2x2 simply made for
                               #positioning and sizing the completed
                               #piece.
        pylab.xticks( pylab.arange(0))
        pylab.yticks( pylab.arange(0))
        # Next, graph the x and z coordinates, with a color and point-type
        #(in this case pixels)
        pylab.plot(xlist,zlist,color=plotcolor[b%7],marker=',',linestyle='None')
        pylab.axis([-20,20,0,50])

    pylab.subplot(2,2,1) 
    for b in range(0,12): #The last file is state11.

        xlist = []
        ylist = []
        zlist = []
        name = data_dir + '/'+base_name + str(b)
        f = open(name, 'r') #Read the data file
        lines = f.readlines()
        # At each line, split the line into the X, Y, and Z coordinates,
        # then add each of them to their own list.
        for line in lines:
            words = string.split(line)
            x = float(words[0])
            y = float(words[1])
            z = float(words[2])
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)        
        pylab.xticks( pylab.arange(0))
        pylab.yticks( pylab.arange(0))
        #graph the x and z coordinates, with a color and point-type (in
        #this case pixels)
        pylab.plot(xlist,zlist,color=plotcolor[b%7],marker=',',linestyle='None')
    pylab.axis([-20,20,0,50])
    pylab.savefig(fig_name) #Make sure to save it as a .pdf
    return 0

if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:

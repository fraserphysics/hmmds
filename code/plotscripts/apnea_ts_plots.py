""" apnea_ts_plots.py a03er_seg a03_lphr a01_lphr a12_lphr a03erA_plot
         a03erN_plot a03HR_plot ApneaNLD

This is the first line of the data file a03er_seg and the fields:

3000.01 -0.65 -4814.0 -4002.0  -56.0     70.0 
time     EEG                   1000*ONR  O2sat

This is the first line of the data file a03.lphr and the fields:

0.0   70.5882352941 2.88250231173
time  lphr          bandpass hr
"""
import matplotlib, sys, os.path, scipy
matplotlib.use('PDF')
import pylab
seg, a01_lphr, a03_lphr, a12_lphr, a03erA_plot, a03erN_plot, a03erHR_plot,\
    ApneaNLD = sys.argv[1:9]

def read_data(data_file):
    # Read in "data_file" as an array
    f = file(data_file, 'r')
    data = [[float(x) for x in line.split()] for line in f.xreadlines()]
    f.close()
    return scipy.array(data).T

seg = read_data(seg)
a01_lphr = read_data(a01_lphr)
a03_lphr = read_data(a03_lphr)
a12_lphr = read_data(a12_lphr)

params = {'axes.labelsize': 12,
               'text.fontsize': 10,
               'legend.fontsize': 10,
               'text.usetex': True,
               'xtick.labelsize': 11,
               'ytick.labelsize': 11}
pylab.rcParams.update(params)

pylab.subplot(2,1,1)
pylab.plot(a01_lphr[0,:],a01_lphr[1,:],'k-')
pylab.ylabel(r'$a01$HR')
yrng = scipy.arange(40,101,20)
pylab.yticks(yrng, [ '$% 3.0f$' % l for l in yrng ])
xrng = range(115,126,5)
pylab.xticks(xrng, [ '$%1d:%02d$' % (l/60,l%60) for l in xrng ])
pylab.axis([115,125,40,100])

pylab.subplot(2,1,2)
pylab.plot(a12_lphr[0,:],a12_lphr[1,:],'k-')
pylab.ylabel(r'$a12$HR')
yrng = scipy.arange(40,81,20)
pylab.yticks(yrng, [ '$% 3.0f$' % l for l in yrng ])
xrng = scipy.arange(570,577,5)
pylab.xticks(xrng, [ '$%1d:%02d$' % (l/60,l%60) for l in xrng ])
pylab.axis([568,577,40,80])

pylab.savefig(ApneaNLD)
pylab.clf() # Clear figure window

pylab.subplot(2,1,1)
pylab.plot(a03_lphr[0,:],a03_lphr[1,:],'k-')
pylab.ylabel(r'$HR$')
yrng = scipy.arange(45,86,10)
pylab.yticks(yrng, [ '$% 2.0f$' % l for l in yrng ])
pylab.xticks([], visible=False)
pylab.axis([55,65,40,90])

pylab.subplot(2,1,2)
pylab.plot(seg[0,:]/60,seg[5,:],'k-')
pylab.ylabel(r'$SpO_2$')
yrng = scipy.arange(60,101,10)
pylab.yticks(yrng, [ '$% 2.0f$' % l for l in yrng ])
xrng = scipy.arange(55,66,5)
pylab.xticks(xrng, [ '$%1d:%02d$' % (l/60,l%60) for l in xrng ])
pylab.axis(xmin=55,xmax=65)

pylab.savefig(a03erHR_plot)
pylab.clf() # Clear figure window
pylab.subplot(3,1,1)
pylab.plot(seg[0,120000:]/60,seg[1,120000:]/1000,'k-')
pylab.ylabel(r'$EEG$')
pylab.xticks([], visible=False)
yrng = scipy.arange(-10,31,10)
pylab.yticks(yrng*1e-3, [ '$% 2.0f$' % l for l in yrng ])
pylab.axis([70,72,-.015,.035])

pylab.subplot(3,1,2)
pylab.plot(seg[0,120000:]/60,seg[4,120000:]/1000,'k-')
pylab.ylabel(r'$ONR$')
pylab.xticks([], visible=False)
yrng = scipy.arange(-10,11,10)
pylab.yticks(yrng, [ '$% 2.0f$' % l for l in yrng ])
pylab.axis([70,72,-15, 15])

pylab.subplot(3,1,3)
pylab.plot(seg[0,120000:]/60,seg[5,120000:],'k-')
pylab.ylabel(r'$SpO_2$')
yrng = scipy.arange(60,101,15)
pylab.yticks(yrng, [ '$% 2.0f$' % l for l in yrng ])
xrng = range(70,73,1)
pylab.xticks(xrng, [ '$%1d:%02d$' % (l/60,l%60) for l in xrng ])
pylab.axis([70, 72, 55, 100])

pylab.savefig(a03erN_plot)
pylab.clf() # Clear figure window

pylab.subplot(3,1,1)
pylab.plot(seg[0,45000:57005]/60,seg[1,45000:57005]/1000,'k-')
pylab.ylabel(r'$EEG$')
pylab.xticks([], visible=False)
yrng = scipy.arange(-10,31,10)
pylab.yticks(yrng*1e-3, [ '$% 2.0f$' % l for l in yrng ])
pylab.axis([57.5, 59.5, -.015, .035])

pylab.subplot(3,1,2)
pylab.plot(seg[0,45000:57005]/60,seg[4,45000:57005]/1000,'k-')
pylab.ylabel(r'$ONR$')
pylab.xticks([], visible=False)
yrng = scipy.arange(-10,11,10)
pylab.yticks(yrng, [ '$% 2.0f$' % l for l in yrng ])
pylab.axis([57.5, 59.5, -15, 15])

pylab.subplot(3,1,3)
pylab.plot(seg[0,45000:57005]/60,seg[5,45000:57005],'k-')
pylab.ylabel(r'$SpO_2$')
yrng = scipy.arange(60,101,15)
pylab.yticks(yrng, [ '$% 2.0f$' % l for l in yrng ])
xrng = range(58,60,1)
pylab.xticks(xrng, [ '$%1d:%02d$' % (l/60,l%60) for l in xrng ])
pylab.axis([57.5, 59.5, 55, 100])

pylab.savefig(a03erA_plot)

""" rr2hr.py Translates r-time anotations to heart rate

python rr2hr.py --Data_Dir=data a01 a02 a03 ... c09 c10

For each record R listed, read R.rtimes, calulate the lowpass filtered
heart rate and write R.lphr.

# Copyright (c) 2005, 2007, 2008 Andrew Fraser
# This file is part of HMM_DS_Code.
#
# HMM_DS_Code is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# HMM_DS_Code is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
1# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

"""
import scipy, numpy.fft as NF, math, sys, getopt
SamPerMin = 10

def record2lphr(record_name, data_dir):
    # Read the rr data (Note that it is sampled irregularly and
    # consists of the r-times in centiseconds) and return high
    # frequency spectrogram.
    Tdata = []
    File = open(data_dir + '/' + record_name + '.Rtimes','r')
    for line in File.xreadlines():
        Tdata.append(float(line)/100)
    File.close()
        # Now Tdata[i] is an r-time in seconds.
    Ddata = scipy.zeros(len(Tdata)-1) # Difference between consecutive Rtimes,
                                  # ie, rr-times
    for i in xrange(len(Ddata)):
        Ddata[i] = Tdata[i+1] - Tdata[i]
    sortedD = scipy.sort(Ddata)
    L = float(len(sortedD))
    ti = int(0.95*L)
    bi = int(0.05*L)
    top = 1.25*sortedD[ti]
    bottom = 0.8*sortedD[bi]
    data = [] # To contain list of pairs [time(in seconds), rr-time]
              # after culling bad rr intervals
    # hack to get an rr time for time zero and first rtime
    rr = Tdata[1] - Tdata[0]
    if rr > top or rr < bottom: # If rr-interval is bad, use median rr-time
        rr = sortedD[int(0.5*len(sortedD))]
    if Tdata[0] > 0:
        data = [[0,rr],[Tdata[0],rr]]
    else:
        data = [[Tdata[0],rr]]
    # Assign good rr times for samples after first and before last
    for i in xrange(len(Tdata)-2):
        rr = Tdata[i+1] - Tdata[i]
        if rr < top and rr > bottom:
            data.append([Tdata[i+1],rr])
    # hack to force rr time for last time
    rr = Tdata[-1] - Tdata[-2]
    if rr > top or rr < bottom:
        rr = sortedD[int(0.5*len(sortedD))]
    data.append([Tdata[-1],rr])

    # Now data[i][0] is an r-time in seconds and data[i][1] is a "good"
    # r-r interval.

    # Create an array of heart rates that is uniformly sampled at 2 HZ

    TF = data[-1][0] # Time is measured in seconds
    L = int(2*TF)
    hr = scipy.zeros(L)
    t_old = data[0][0]
    rr_old = data[0][1]
    t_new = data[1][0]
    rr_new = data[1][1]
    i=1
    for k in xrange(L):
        t = k/2.0
        while t > t_new:
            i += 1
            if i >= len(data):
                break
            t_old = t_new
            rr_old = rr_new
            t_new = data[i][0]
            rr_new = data[i][1]
        if i >= len(data):
            break
        drdt = float(rr_new-rr_old)/float(t_new-t_old)
        rr =  rr_old + (t-t_old)*drdt
        hr[k] = rr
    # Transform rr_times to heart rate
    hr = 60/(hr)
    Avg = hr.sum()/L
    hrL = hr - Avg
    # Now, hr is heart rate sampled at 2HZ, and hrL is the same with
    # mean subtracted

    HR = NF.rfft(hrL,131072) # 131072 is 18.2 Hrs at 2HZ
    HR[0:100] *=0 # Drop frequencies below (100*60)/65536=0.09/min
    HR[4000:] *=0 # and above (4000*60)/65536=3.66/min
    hrL = NF.irfft(HR)
    File = open(data_dir + '/' + record_name + '.lphr','w')
    for i in xrange(0,L,(2*60)/SamPerMin):
        print >>File, i/120.0, hr[i], hrL[i]
    # Time in minutes, Unfiltered hr in beats per minute, low pass hr
    File.close()

opts,pargs = getopt.getopt(sys.argv[1:],'',['Data_Dir='])
if len(opts) == 1:
    if opts[0][0] == '--Data_Dir':
        data_dir = opts[0][1]
    else:
        raise RuntimeError,'bad keyword argument %s'%opts[0][0]
else:
    data_dir = 'data/Apnea'
# Process each record listed on the command line
for name in pargs:
    record2lphr(name,data_dir)

#Local Variables:
#mode:python
#End:

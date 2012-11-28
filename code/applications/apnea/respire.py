""" respire.py

python respire.py --Data_Dir=data --Annotations=data/summary_of_training\
 --ApOb_Dir=code/hmm/ApOb.py a01 a02 a03 ... x34 x35

Calculate high frequency periodograms for each record at 0.1 minute
intervals.  Collect these vectors into three groups:

1. Those from 'c' records
2. Those from 'a' records during normal sleep
3. Those from 'a' records during apnea

From these three groups, calculate two Fisher LDA basis vectors.

For each sample time for each record, project the corresponding vector
onto the Fisher basis and write the result to data/'record'.resp2

Copyright (c) 2005, 2008 Andrew Fraser
This file is part of HMM_DS_Code.

HMM_DS_Code is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

HMM_DS_Code is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc.,
59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

"""
import scipy, scipy.linalg, math, sys, cinc2000, getopt, numpy.fft

SamPerMin = 10               # Samples per minute for output
Dt_in = 0.5                  # Sampling interval of jitter in seconds
SpM = 60                     # Seconds per minute
Dt_out = SpM/SamPerMin       # Output sampling interval in seconds
RDt = int(Dt_out/Dt_in)      # Ratio of sampling intervals
Fw = 1024                    # Channels for FFT
cpm_chan = SpM/(Dt_in*Fw)    # (cycles per minute)/(channel #)
Glength = 1024               # Length of data sequence in samples
Gl_2 = Glength/2
CUT = 1.0                    # Exclude high magnitude samples from
                             # basis function calculation
sigma = 50.0                 # Width of Gaussian window in samples.
                             # Easier to change sigma than Fw, etc

# Calculate Gaussian window function
Gw = scipy.zeros((Glength,),scipy.float64)
for t in xrange(Glength):
    Gw[t] = math.exp( -((t-Gl_2)**2)/(2*sigma**2) )

def record2vecs(record_name,data_dir):
    # Read the rr data (Note that it is sampled irregularly and
    # consists of the r-times in centiseconds) and return a high
    # frequency spectrogram.
    name = data_dir + '/' + record_name + '.Rtimes'
    data = []
    File = open(name,'r')
    for line in File.xreadlines():
        data.append(float(line)/100)
    File.close()
    # Now data[i] is an r-time in seconds.
    hrd = cinc2000.R_times2Dev(data)
    # hrd are heart rate deviations sampled at 2 Hz
    pad = scipy.zeros(((Glength+len(hrd)),),scipy.float64)
    pad[Gl_2:len(hrd)+Gl_2] = hrd
    # Now pad is hrd with Gl_2 zeros before and after
    N_out = len(hrd)/RDt
    result = scipy.zeros((N_out,Fw/2),scipy.float64)
    mags = []
    for k in xrange(N_out):
        i = int(RDt*k)
        WD = Gw*pad[i:i+Glength] # Multiply data by window fuction
        FT = numpy.fft.rfft(WD,n=Fw)
        SP = scipy.conjugate(FT)*FT  # Periodogram
        temp = numpy.fft.rfft(SP,n=Fw/2)
        SP = numpy.fft.irfft(temp[0:int(0.1*Fw/2)],n=Fw/2)
        # Low pass filter in freq domain.  Pass below 0.1 Nyquist
        temp = SP.real[:Fw/2]
        mag = math.sqrt(scipy.dot(temp,temp))
        result[k,:] = temp/mag
        mags.append(math.log(mag))
        # result[k,:] is a unit vector and a smoothed periodogram
    return [result,mags]

opts,pargs = getopt.getopt(sys.argv[1:],'',[
        'Annotations=',
        'Data_Dir=',
        'ApOb_Dir='
        ])
opt_dict ={}
for opt in opts:
    if len(opt) is 2:
        opt_dict[opt[0]] = opt[1]
    else:
        opt_dict[opt[0]] = True 
Annotations = opt_dict['--Annotations']
Data_Dir = opt_dict['--Data_Dir']
ApDat = Data_Dir+'/Apnea'
ApOb_Dir = opt_dict['--ApOb_Dir']
sys.path.append(ApOb_Dir)
import ApOb

C_vecs  = [] # Vectors from c records
AA_vecs = [] # Vectors from a records in apnea time
AN_vecs = [] # Vectors from a records in normal time
vec_dict = {}

# Read data from the records listed on the command line
for name in pargs:
    vecs,mags = record2vecs(name,ApDat)
    vec_dict[name] = [vecs,mags]
    if name[0] == 'b' or name[0] == 'x': # Not used for Fisher LDA
        continue
    if name[0] == 'c':
        for vec in vecs[300:-60]:
            C_vecs.append(vec)
        continue
    assert (name[0] is 'a'),'record %s is not a*, b*, c*, or x*'%name
    Ap_notes = ApOb.fetch_annotations(Annotations,name)
    # Now Ap_notes elements have form [time(minutes),Mark] where
    # Mark=1 for Apnea and Mark=0 for Normal
    #for t in xrange(min(len(vecs),SamPerMin*len(Ap_notes))):
    for t in xrange(300,len(vecs)-300):
        M = int(t/SamPerMin)
        if mags[t] > CUT:
            continue
        if Ap_notes[M] == 1:
            AA_vecs.append(vecs[t])
        elif Ap_notes[M] == 0:
            AN_vecs.append(vecs[t])
        else:
            raise RuntimeError,'For %s, Ap_notes[%d][1]=%d'%(name, M,
                                                             Ap_notes[M][1])
# End of loop for reading records

# Calculate mean and scatter of the three classes
def mean_var(IN):
    vecs = scipy.array(IN)
    mean = scipy.sum(vecs,axis=0)/len(vecs)
    d = scipy.transpose(vecs-mean)
    var = scipy.inner(d,d)
    return [mean,var,vecs,len(vecs)]
C_mean,C_var,C_vecs,C_n = mean_var(C_vecs)
AA_mean,AA_var,AA_vecs,AA_n = mean_var(AA_vecs)
AN_mean,AN_var,AN_vecs,AN_n = mean_var(AN_vecs)

print 'C_mean.shape=',C_mean.shape, 'C_var.shape=',C_var.shape, 'C_vecs.shape=',C_vecs.shape # FixMe: Remove this later

# Calculate Sw, the within class scatter
Sw = C_var + AA_var + AN_var

# Calculate Sb, the between class scatter
n = C_n + AA_n + AN_n
mean = (C_n*C_mean + AA_n*AA_mean + AN_n*AN_mean)/n
def Sb_t(tmean,tn,mean):
    d = tmean-mean
    return tn*scipy.outer(d,d)
Sb = Sb_t(C_mean,C_n,mean)+Sb_t(AA_mean,AA_n,mean)+Sb_t(AN_mean,AN_n,mean)

# Calculate a 2-d basis of linear discriminant vectors
n = Sw.shape[0]
vals,vecs = scipy.linalg.eigh(scipy.dot(scipy.linalg.inv(
                        Sw+scipy.eye(n)*100),Sb))
# Find largest two eigenvalues
L = vals.tolist()
i0 = L.index(max(vals))
temp = vals[i0]
vals[i0] = 0
i1 = L.index(max(vals))
vals[i0] = temp
basis = scipy.zeros((n,2))
basis[:,0] = vecs[:,i0]
basis[:,1] = vecs[:,i1]

# Write projections of the data to 'record.resp'
for key in vec_dict.keys():
    vecs,mags = vec_dict[key]
    pairs = scipy.dot(vecs,basis)
    name = ApDat + '/' + key + '.resp'
    File = open(name,'w')
    for i in xrange(len(pairs)):
        print >>File,i/float(SamPerMin),pairs[i,0],pairs[i,1],mags[i]

# Write files of information to characterize LDA
File = open(Data_Dir+'/mean.resp','w')
for i in xrange(len(C_mean)):
    print >>File, i, C_mean[i], AN_mean[i], AA_mean[i], basis[i,0], basis[i,1]

for name,data in [['C.resp',C_vecs],['AN.resp',AN_vecs],['AA.resp',AA_vecs],]:
    dots = scipy.dot(data,basis)
    File = open(Data_Dir+'/'+name,'w')
    for i in xrange(dots.shape[0]):
        print >>File, dots[i,0], dots[i,1]
    File.close()

# Old stuff for matplotlib
# from pylab import *
# figure(1)
# plot(vals,'r-')
# wvals,wvecs = eigh(Sw)
# plot(wvals,'b-')


# # Plot class averages and basis vectors
# figure(2)
# plot(basis[:,0],'r--')
# plot(basis[:,1],'b--')
# plot(C_mean[:],'r-')
# plot(AA_mean[:],'g-')
# plot(AN_mean[:],'b-')

# # Make scatter plot of the three classes
# figure(3)
# def plot_dots(vecs,basis,style):
#     dots = scipy.dot(vecs,basis)
#     plot(dots[:,0],dots[:,1],style)
# plot_dots(C_vecs,basis,'r,')
# plot_dots(AA_vecs,basis,'g,')
# plot_dots(AN_vecs,basis,'b,')
# figure(4)
# plot_dots(C_vecs,basis,'r,')
# figure(5)
# plot_dots(AA_vecs,basis,'g,')
# figure(6)
# plot_dots(AN_vecs,basis,'b,')
# show()

#Local Variables:
#mode:python
#End:

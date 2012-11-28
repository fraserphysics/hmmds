""" ApOb.py

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

The code in this file implements the apnea observation model class
ApObModel which models the scalar time series of low frequency heart
rate variability via Gaussians with affine autoregressive means and
the high frequency variability via a 2-d _respiration_ model that fits
the power and maximum frequencey in a high-pass band.

For training, the observation data y[t] has the form

y[t][0] The scalar observation
y[t][1] The context vector
y[t][2] The classification (an integer)

For testing, observation data y[t] has the form

y[t][0] The scalar observation
y[t][1] The context vector

When training, if y[t][2] is not equal to Class, the probability is
set to zero.

"""

small = 1e-25
big = 1e+25
SamPerMin = 10 # Samples per minute.

import Scalar, EXT, numpy, numpy.linalg as LA, numpy.random
import pickle, math, random, copy
LAI = LA.inv

Mark_dict = {'N':0,'A':1}
def fetch_ann(Annotations,name):
    """ Like fetch_annotations, but shorter result.  Only one sample per
    minute.
    """
    F = open(Annotations,'r')
    parts = F.readline().split()
    while len(parts) is 0 or parts[0] != name:
        parts = F.readline().split()
    hour = 0
    letters = []
    for line in F.xreadlines():
        parts = line.split()
        if len(parts) != 2:
            break
        assert (int(parts[0]) == hour),"hour wrong"
        hour += 1
        letters += parts[1]
    notes = []
    for t in xrange(len(letters)):
        notes.append(Mark_dict[letters[t]])
    return numpy.array(notes)
def fetch_annotations(Annotations,name):
    return fetch_ann(Annotations,name).repeat(SamPerMin) 

def read_data(data_file):
    # Read in "data_file" as a 2-d array
    f = file(data_file, 'r')
    data = [[float(x) for x in line.split()] for line in f.xreadlines()]
    return numpy.array(data).T

def read_lphr(where,what,AR):
    """ Create numpy array suitable for HR_HMM.Py_wo_class.
    y[t,0]      heart rate
    y[t,1:AR+1] previous AR heart rates
    y[t,AR+1]   constant 1.0
    """
    filename = where%what
    raw = numpy.flipud(read_data(filename)[1])
    #raw = read_data(filename)[1]
    T = len(raw)
    Y = numpy.empty((T,AR+2))
    for t in xrange(AR+2):
        Y[t,:AR+1] = raw[-AR-1:]
    for t in xrange(AR+2,T):
        Y[t,:AR+1] = raw[-t:AR+1-t]
    raw.sort
    scale = raw[int(T*.8)]*2.0
    Y *= scale
    Y[:,-1] = 1
    return Y    

def read_resp(where,what):
    """ Create numpy array suitable for Resp_HMM.Py_wo_class.
    """
    return read_data(where%what)[1:].T

def read_records(routines, # List of routines to read data
                 paths,    # List of strings pointing to data
                 args,     # Arg that is either AR or None
                 records   # List of records to process, eg, ['a01','a02',...]
                 ):
    """ Read the records specified.  Return "Ys" with the form like
    [[HR_record_0, resp_record_0],[HR_record_1, resp_record_1], ...]
    Return "Yall" with form like [cat(HR_record_0,HR_record_1,...),
    cat(resp_record_0,resp_record_1,...)]
    """
    Ys = []
    Yall = []
    for record in records:
        dats = []
        for routine,path,arg in zip(routines,paths,args):
            if arg is None:
                dats.append(routine(path,record))
            else:
                dats.append(routine(path,record,arg))
        T = min(map(lambda dat:len(dat), dats))
        dats = map(lambda dat:dat[:T], dats)
        if record is records[0]:
            Ys = [dats] 
            Yall = map(lambda (dat):numpy.array(dat,copy=True), dats)
        else:
            Ys.append(dats)
            Yall = map(
                 lambda (all,dat):numpy.concatenate((all,dat)),zip(Yall,dats))
    return (Ys,Yall)

def score(reference,test,records,verbose=False):
    """ Find fraction of classifications in "test" that match
    classifications in "reference".
    """
    def report(reference,test,record):
        A = numpy.array(fetch_ann(reference,record),numpy.bool)
        B = numpy.array(fetch_ann(test,record),numpy.bool)
        T = min(len(A),len(B))
        A = A[:T]
        B = B[:T]
        NN = ((-A)*(-B)).sum()  # Number of normals classified as normal
        FA = ((-A)*B).sum()     # Number of false alarms
        MD = (A*(-B)).sum()     # Number of missed detections
        AA = (A*B).sum()        # Number of true alarms
        return numpy.array([NN,FA,MD,AA])
    if verbose:
        rv = 'rec  N->N N->A A->N  A->A   frac\n'
        total = numpy.zeros(4,numpy.int32)
        for record in records:
            x = report(reference,test,record)
            total += x
            rv += '%s %5d %4d %4d %5d  %5.3f\n'%(record,x[0],x[1],x[2],x[3],
                                            float(x[0]+x[3])/float(x.sum()))
        rv += '    %5d %4d %4d %5d  %5.3f\n'%(total[0],total[1],total[2],
                total[3],float(total[0]+total[3])/float(total.sum()))
        return rv
    else:
        total = numpy.zeros(4,numpy.int32)
        for record in records:
            total += report(reference,test,record)
        return total

def doctor_M(M,Fudge):
    """ Force system to start in state 0.  Multiply output probabilities
    of all states in class A by Fudge and multiply transitions between A
    and N by 0.001.
    """
    M.P_S0 *=0
    M.P_S0[0,0] = 1.0
    M.P_S0_ergodic *=0
    M.P_S0_ergodic[0,0] = 1.0
    for i in xrange(M.N):
        for j in xrange(M.N):
            if M.S2C[i] != M.S2C[j]:
                M.P_ScS[i,j] *= 1e-6
        if not(Fudge is None) and  M.S2C[i] == 0:
            M.HR.norm[i] *= Fudge
            M.Resp.norm[i] *= Fudge
    M.P_ScS /= M.P_ScS.sum(axis=1)
    return # End of doctor_M()

def read_mod(name,fudge,Pow):
    fudge = float(fudge)
    Pow = float(Pow)
    mod = pickle.load(open(name, 'r'))
    mod.Pow = Pow
    if fudge > 0:
        doctor_M(mod,fudge)
    return mod # End of read_mod()

######### Begin model that stores class as separate array ##########
class BASE(EXT.HMM):
    """ Common base for HR_HMM, Resp_HMM, and SB_HMM.  Key feature is that
    observations are lists of time series rather than time series of
    lists, eg for SB_HMM Y is stored as [C,HR,Resp].  Since Y[t]
    doesn't make sense, I redefine the following methods: join_Ys(),
    PY_w_class() and reestimateC().

    Don't use class BASE directly; use it as a base for subclasses.
    """
    def Py_w_class(self,Y):
        Py = self.Py_wo_class(Y[1:])
        for t in xrange(self.T):
            C = Y[0][t]
            Py_t = numpy.zeros(self.N,numpy.float64)
            for s in self.C2S[C]:
                Py_t[s] = Py[t,s]
            Py[t,:] = Py_t
        return Py
    def reestimateC(self,Y):
        self.reestimate_noC(Y[1:])
        return
    def join_Ys(self, # Both_HMM
                Ys    # List of segments
                ):
        """ Ys has a form like the following:
        [
          [C,HR,Resp] # For first record
          ...
          [C,HR,Resp] # For last record
        ]
        """
        N = len(Ys[0]) # Number of components, eg 3 for [C,HR,Resp]
        Y_all = []     # Collection of N time series
        Tseg = [0,len(Ys[0][0])] # List of segment boundaries
        for i in xrange(N):
            Y_all.append(numpy.array(Ys[0][i],copy=True))
        for Y in Ys[1:]:
            assert len(Y) is N
            Tseg.append(len(Y[0])+Tseg[-1])
            for i in xrange(N):
                Y_all[i] = numpy.concatenate((
                          Y_all[i], numpy.array(Y[i],copy=False)))
        self.T = len(Y_all[-1])
        return self.Py_calc(Y_all),Y_all,Tseg,len(Ys)
############## Begin model for respiration only ####################
class Resp_HMM(BASE):
    """ Observation model for respiration signal.
    y[0][t] = respiration vector at time t
    y[1][t] = Class if y[1] exists    
    
    """
    def __init__(self, P_S0,P_S0_ergodic,P_ScS,C2S,mu,Icov,norm):
        BASE.__init__(self,P_S0,P_S0_ergodic,P_ScS,P_YcS=None,C2S=C2S)
        self.mu=numpy.array(mu)
        self.Icov=numpy.array(Icov)
        self.norm=numpy.array(norm)
        self.Py_calc = None # Assign to Py_wo_class or Py_wo_class
        self.reestimate = None # Assign to reestimate_noC or reestimateC
    def dump(self # Resp_HMM
             ):
        self.dump_base()
        for i in xrange(self.N):
            Icov = self.Icov[i]
            print ' For state %d:'%i
            Scalar.print_Name_VV('Icov',Icov)
            print '              mu  =',self.mu[i]
            print '              norm =',self.norm[i]
        return #end of dump()
    def randomize(self # Resp_HMM
             ):
        """ Perturb the observation models for each state to break symmetry.
        For each state i, draw a random number from N(0,cov[i]) and
        add it to mu[i].
        """
        for i in xrange(self.N):
            cov = LAI(self.Icov[i])
            self.mu[i] = numpy.random.multivariate_normal(self.mu[i],cov)
    def Py_wo_class(self # Resp_HMM
             ,YL):
        """ Caclculate observation probabilities without Class.  Called by
        class_decode and Py_w_class
        """
        y = YL[0]
        # Check size and initialize self.Py
        self.T = len(y)
        try:
            assert(self.Py.shape is (self.T,self.N))
        except:
            self.Py = numpy.zeros((self.T,self.N),numpy.float64)
        for t in xrange(self.T):
            for i in xrange(self.N):
                d = numpy.mat(y[t]-self.mu[i]).T
                dQd = d.T*numpy.mat(self.Icov[i])*d
                dQd = min(float(dQd),300.0) # Underflow
                self.Py[t,i] = self.norm[i]*math.exp(-dQd/2)
        return self.Py # End of Py_wo_class
    def reestimate_noC(self, #Resp_HMM
                       YL,
                       O_mod_only=False):
        """ Reestimate parameters using a sequence of observations without
        Class data.
        """
        Y = YL[0]
        if not O_mod_only:
            self.reestimate_s() # Reestimate all except observation models
        Y = numpy.array(Y,copy=False)  # TxDim array
        T,Dim = Y.shape
        assert Dim == 3
        assert T > 50
        w = self.alpha*self.beta  # w[t,i] = prob s(t) = i
        wsum = w.sum(axis=0)
        self.mu = (numpy.inner(Y.T,w.T)/wsum).T
        # Inverse Wishart prior parameters.  Without data sigma_sq = b/a
        a = 4
        b = 0.1
        for i in xrange(self.N):
            rrsum = numpy.mat(numpy.zeros((Dim,Dim)))
            for t in xrange(len(Y)):
                r = Y[t]-self.mu[i]
                rrsum += w[t,i]*numpy.outer(r,r)
            cov = (b*numpy.eye(Dim) + rrsum)/(a + wsum[i])
            det = LA.det(cov)
            assert(det) > 0.0
            assert (LA.det(cov) > 0.0)
            self.Icov[i,:,:] = LAI(cov)
            self.norm[i] = 1.0/(math.sqrt((2*math.pi)**Dim*det))
# End of class Resp_HMM
############## Begin model for heart rate only ####################
class HR_HMM(BASE):
    """ Has autoregressive observation model for heart rate signal.
    y[0][t][0] = hr
    y[0][t][1:] = context
    y[1][t] = Class if y[1] exists
    
    """
    def __init__(self, P_S0,P_S0_ergodic,P_ScS,C2S,A,Var,norm):
        BASE.__init__(self,P_S0,P_S0_ergodic,P_ScS,P_YcS=None,C2S=C2S)
        self.A=numpy.array(A)
        self.Var=numpy.array(Var)
        self.norm=numpy.array(norm)
        self.Py_calc = None # Assign to Py_wo_class or Py_wo_class
        self.reestimate = None # Assign to reestimate_noC or reestimateC
    def dump(self # HR_HMM
             ):
        self.dump_base()
        for i in xrange(self.N):
            A = self.A[i]
            print ' For state %d:'%i
            Scalar.print_Name_V(
                  '              A   ',A)
            print '              Var  =',self.Var[i]
            print '              norm =',self.norm[i]
        return #end of dump()
    def randomize(self):
        """ Perturb the observation models for each state to break symmetry.
        For each state i, draw a random number from N(0,Var[i]) and
        add it to A[i,-1].
        """
        for i in xrange(self.N):
            self.A[i,-1] += random.gauss(0,math.sqrt(self.Var[i]))
    def Py_wo_class(self, # HR_HMM
                    YL):
        """ Caclculate observation probabilities without Class.  Called by
        class_decode and Py_w_class.  y[t,0] is hr and y[t,1:] is context
        """
        y = YL[0]
        # Check size and initialize self.Py
        self.T = len(y)
        y = numpy.array(y,copy=False)
        try:
            hr = y[:,0].reshape((self.T,))
        except:
            print 'y=',y
            hr = y[:,0].reshape((self.T,))
        context = y[:,1:]
        d =  hr - numpy.inner(self.A,context)
        # d[i,t] is hr[t] - A[i] dot context[t]
        try:
            assert(self.Py.shape is (self.T,self.N))
        except:
            self.Py = numpy.zeros((self.T,self.N),numpy.float64)
        for i in xrange(self.N):
            z = numpy.minimum(d[i]*d[i]/(2*self.Var[i]),300.0)
            # Cap z to stop underflow
            self.Py[:,i] = numpy.exp(-z)*self.norm[i]
        return self.Py
    def reestimate_noC(self, #HR_HMM
                       YL,
                       O_mod_only=False):
        """ Reestimate parameters using a sequence of observations without
        Class data.
        """
        Y = YL[0]
        if not O_mod_only:
            self.reestimate_s() # Reestimate all except observation models
        Y = numpy.array(Y,copy=False)  # TxDim array
        T,Dim = Y.shape
        w2 = self.alpha*self.beta # w2[t,i] = prob s(t) = i
        mask = w2 >= small        # Small weights confuse the residual
                                  # calculation in least_squares()
        w2 *= mask
        wsum = w2.sum(axis=0)
        w = numpy.sqrt(w2)      # TxN array of weights
        # Inverse Wishart prior parameters.  Without data, sigma = b/a
        a = 4
        b = 16
        for i in xrange(self.N):
            HR = w.T[i]*Y.T[0]           # Tx1
            context = (w.T[i]*Y.T[1:]).T # Tx(Dim-1)
            A,resids,rank,s = LA.lstsq(context,HR)
            z = HR-numpy.inner(context,A) # z[t] is a number
            zz = float(numpy.inner(z,z))  # zz is a number
            self.Var[i] = (b+zz)/(a+wsum[i])
            self.A[i,:] = A
            self.norm[i] = 1/math.sqrt(2*math.pi*self.Var[i])

############## Begin model for both respiration and lphr ####################
class Both_HMM(BASE):
    """ Combines Resp_HMM and HR_HMM.
    
    """
    def __init__(self, P_S0,P_S0_ergodic,P_ScS,C2S,mu,Icov,normR,A,Var,normH,
                 Pow=1):
        BASE.__init__(self,P_S0,P_S0_ergodic,P_ScS,P_YcS=None,C2S=C2S)
        self.Resp = Resp_HMM(P_S0,P_S0_ergodic,P_ScS,C2S,mu,Icov,normR)
        self.HR = HR_HMM(P_S0,P_S0_ergodic,P_ScS,C2S,A,Var,normH)
        self.Pow = Pow
        self.Py_calc = None # Assign to Py_wo_class or Py_wo_class
        self.reestimate = None # Assign to reestimate_noC or reestimateC
    def dump(self # Both_HMM
             ):
        self.dump_base()
        print "S2C=",self.S2C
        for i in xrange(self.N):
            print ' For state %d:'%i
            Icov = self.Resp.Icov[i]
            Scalar.print_Name_VV(' Icov',Icov)
            print '              mu    =',self.Resp.mu[i]
            print '              normR =',self.Resp.norm[i]
            A = self.HR.A[i]
            Scalar.print_Name_V(
                  '              A    ',A)
            print '              Var   =',self.HR.Var[i]
            print '              normH =',self.HR.norm[i]
        return #end of dump()
    def randomize(self # Both_HMM
             ):
        """ Breaks symmetry of identical states with plausible shift of mean.
        """
        self.Resp.randomize()
        return
    def Py_wo_class(self # both_HMM
             ,YL):
        pyH = self.HR.Py_wo_class([YL[0]])
        pyR = self.Resp.Py_wo_class([YL[1]])
        self.Py = pyR * (numpy.power(pyH,self.Pow))
        self.T = len(self.Py)
        return self.Py
    def reestimate_noC(self, #Both_HMM
                       YL):
        self.reestimate_s() # Reestimate all except observation models
        self.HR.alpha = self.alpha
        self.HR.beta = self.beta
        self.HR.gamma = self.gamma
        self.HR.Py = self.Py
        self.Resp.alpha = self.alpha
        self.Resp.beta = self.beta
        self.Resp.gamma = self.gamma
        self.Resp.Py = self.Py
        self.HR.reestimate_noC([YL[0]],O_mod_only=True)
        self.Resp.reestimate_noC([YL[1]],O_mod_only=True)
        return
    # End of Both_HMM
############## Begin structured model for both HR and Resp ##################
class SB_HMM(Both_HMM):
    """ Like Both_HMM, but S2C list is replaced by list of dicts.  Py[t,s]
    is masked to zero if not S2C[s].has_key(y[t,0])
    
    """
    def __init__(self, P_S0,P_S0_ergodic,P_ScS,C2S,mu,Icov,normR,A,Var,normH,
                 Pow=1):
        Both_HMM.__init__(self, P_S0, P_S0_ergodic, P_ScS, C2S, mu, Icov,
                          normR, A, Var, normH, Pow=Pow)
        S2C_dict = self.N*[None]
        for s in xrange(self.N):
            S2C_dict[s] = {self.S2C[s]:True}
        self.S2C = S2C_dict
    def Py_w_class(self,   # SB_HMM
                   YL):
        """ Y is stored as [C,HR,Resp].  Py[t,i] = 0 unless
        S2C[i].has_key(C[t]).
        """
        Py = self.Py_wo_class(YL[1:])
        for t in xrange(self.T):
            C = YL[0][t]
            Py_t = numpy.zeros(self.N,numpy.float64)
            for s in xrange(self.N):
                if self.S2C[s].has_key(C):
                    Py_t[s] = Py[t,s]
            Py[t,:] = Py_t
        return Py
    def add_class(self,    # SB_HMM
                  state,
                  Class):
        self.S2C[state][Class] = True
    def make_cluster(self, # SB_HMM
                     home, # The number of the "home" state
                     N_isles, N_peaks, peak_class):
        """ Use the sequence of states [home:home+1+N_isles+2*N_peaks] to make
        a structured cluster.  Each "isle" is connected from and to
        "home".  Each "peak" is a two state chain connected to "home".
        The distal state of each chain gets an additional class that
        permits it to model "peak" observations.  Each "peak"
        observation must fall in such a state.
        """
        for i in xrange(home+1,home+N_isles+1):
            self.link(home,i,.1)
            self.link(i,home,.1)
        for i in xrange(home+N_isles+1,home+N_isles+1+N_peaks*2,2):
            self.link(home,i,.1)
            self.link(i,home,.1)
            self.link(i+1,i,.1)
            self.link(i,i+1,.1)
            self.add_class(i+1,peak_class)
            
    # End of SB_HMM

############## Begin structured model for HR alone ##################
class SH_HMM(HR_HMM,SB_HMM):
    """ Like SB_HMM, but Y is [C,H] only
    
    """
    def __init__(self, P_S0,P_S0_ergodic,P_ScS,C2S,A,Var,norm):
        HR_HMM.__init__(self, P_S0, P_S0_ergodic, P_ScS, C2S, A, Var, norm)
        S2C_dict = self.N*[None]
        for s in xrange(self.N):
            S2C_dict[s] = {self.S2C[s]:True}
        self.S2C = S2C_dict
    def Py_w_class(self,YL):
        return SB_HMM.Py_w_class(self,YL)
    def randomize(self # SH_HMM
             ):
        """ Breaks symmetry of identical states with plausible shift of mean.
        """
        HR_HMM.randomize(self)
        return
#Local Variables:
#mode:python
#End:

'''mod_init.py script to create initial models for work on the
CINC2000 data.

Argument     Usual Value

HR_dir       derived_data/apnea/low_pass_heart_rate 

Resp_Dir     derived_data/apnea/respiration

Expert       raw_data/apnea/summary_of_training

Model_Dir    derived_data/apnea

This script makes the following models for internal use:

base_s1_ar2:    One state, AR order 2, fit to all of the data

base_s1_resp:   One state, Resp model, fit to all of the data

base_s1_ar4:    One state, AR order 4, fit to all of the data

base_n1_a1_ar2: One normal state and one apnea state, fit to all of
                the data and use classification
base_n1_a1_ar4: One normal state and one apnea state, fit to all of
                the data and use classification

This script writes the following five models to "Model_Dir"

init_A2:        Two state, no class, AR=4.

mod_C1:         One state, no class, AR-4.  Make with one pass in this script

init_H:         Two class, topology in Fig 6.9a. Resp and AR=2

init_M:         Two class, topology in Fig 6.9b. Resp and AR=4

init_L:         Two class, topology in Fig 6.9b. Resp and AR=4 

Everything is hard coded with "magic" numbers.  From the text, I copy
the following specifications:

mod_A2: A two state HMM with AR-4 models for the lprh (low pass heart
        rate) data and Gaussian models for resp (the respiration
        data).  No classification used.

mod_C1: A one state model with AR-4 for lphr and a single Gaussian for
        resp

In the three models below, each state is a member of a class.

mod_H: Topology in Fig 6.9a.  Used if first pass stat is above 3.0.
AR order = 2

mod_M: Topology in Fig 6.9b.  Used if first pass stat is between 2.39
       and 3.0 AR order 4.

mod_L: Topology in Fig 6.9b.  Used if first pass stat is below 2.39.
       AR order 4.

Found AR orders by looking in hmmdsbook/code/Apnea/Makefile.  Also
found the following three lists:
'''

LOW  = '''c04 c09 c03 c10 c02 c06
          c05 c01 c07 c08 b04'''

MED  = '''c07 c08 b04 a06 b01 a11'''

HIGH = '''a06 b01 a11 a10 a17 a18
          a05 b03 a16 a20 a09 a14
          a07 a19 b02 a02 a08 a13
          a03 a01 a04 a12 a15'''

import sys
def main(argv=None):
    '''Call with arguments: 

    '''

    if argv is None:                    # Usual case
        argv = sys.argv[1:]
    assert len(argv) == 4
    hr_dir, resp_dir, expert, model_dir = argv

    import os
    import numpy as np
    from base import HMM
    import Scalar
    import ApOb
    # Set up lists of records by the groups used for training
    all_records = os.listdir(hr_dir)
    a_records = [x for x in all_records if x.startswith('a')]
    b_records = [x for x in all_records if x.startswith('b')]
    c_records = [x for x in all_records if x.startswith('c')]
    low_records = LOW.split()
    med_records = MED.split()
    high_records = HIGH.split()

    # Create models that have each of the kinds of observation model
    # that I will use later.  If the observation model has
    # classification, the HMM has two states and if not, it has only
    # one state.  These models train in a single pass through the
    # data.
    P_SS = np.ones((1,1,1), np.float64)
    P_S0 = np.ones((1,1), np.float64)
    Var = np.empty((1,1))
    norm = np.empty((1,1))
    y_ar2 = (np.empty((1,3)), Var, norm)
    y_ar4 = (np.empty((1,5)), Var, norm)
    mu = np.empty((1,3))
    Icov = np.empty((1,3,3))
    y_resp = (mu, Icov, norm)
    y_both = lambda x : ( ApOb.Both,
                          (
                              x,         # hr_params
                              y_resp     # resp_params
                          ),
                          {0:[0], 1:[1]} # c2s
                      )
    base_s1_resp = HMM(P_S0, P_S0, y_resp, P_SS, ApOb.Resp)
    base_s1_ar2 = HMM(P_S0, P_S0, y_ar2, P_SS, ApOb.Heart_Rate)
    base_s1_ar4 = HMM(P_S0, P_S0, y_ar4, P_SS, ApOb.Heart_Rate)
    base_n1_a1_ar2_resp = HMM(P_S0, P_S0, y_both(y_ar2), P_SS, Scalar.Class_y)
    base_n1_a1_ar4_resp = HMM(P_S0, P_S0, y_both(y_ar4), P_SS, Scalar.Class_y)
    for mod in (base_n1_a1_ar2_resp, base_n1_a1_ar4_resp):
        mod.y_mod.set_dtype([np.int32, np.float64, np.float64, np.float64])

    # Class for args instances for reading data
    class ARGS:
        def __init__(self, record, expert=None, resp_dir=None, hr_dir=None):
            self.record = record
            self.expert = expert
            self.resp_dir = resp_dir
            self.hr_dir = hr_dir
            return
        def __str__(self):
            return('expert=%s, resp_dir=%s hr_dir=%s\nrecord=%s'%(
                self.expert, self.resp_dir, self.hr_dir, self.record))

    # Train each of the five base models with a single pass through
    # the appropriate data
    marked = a_records + b_records + c_records
    for model, args in (
            (base_n1_a1_ar4_resp,
             ARGS(marked, hr_dir=hr_dir, resp_dir=resp_dir,
                  expert=expert)),
            (base_n1_a1_ar4_resp,
             ARGS(marked, hr_dir=hr_dir, resp_dir=resp_dir,
                  expert=expert)),
            (base_s1_ar2, ARGS(all_records, hr_dir=hr_dir)),
            (base_s1_ar4, ARGS(all_records, hr_dir=hr_dir)),
            (base_s1_resp, ARGS(all_records, resp_dir=resp_dir)),
          ):
        y_dict = ApOb.build_data(model.y_mod, args)
        n_seg, segs, data = model.y_mod.join(list(y_dict.values()))
        n_y = len(data[0])
        n_states = model.n_states
        if n_states == 1:
            w = np.ones((n_y, 1))
        else:
            w = np.empty((n_y, 2))
            w[:,0] = data[0]
            w[:,1] = 1 - data[0]
        model.y_mod.reestimate(w, data)

    return 0

if __name__ == "__main__":
    #hr_dir, resp_dir, expert, model_dir
    argv = (
        '../../../derived_data/apnea/low_pass_heart_rate',
        '../../../derived_data/apnea/respiration',
        '../../../raw_data/apnea/summary_of_training',
        '../../../derived_data/apnea/'
        )
    sys.exit(main(argv))
    
#Local Variables:
#mode:python
#End:

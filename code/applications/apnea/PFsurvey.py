""" DoubleClassify.py command_options list_of_data_files

EG: python DoubleClassify.py --Model=mod --Annotations=reference
--Dpath=data/Apnea --Power 0.5 2.65 0.1 --Fudge .8 1.61 .05 a06 b01 ...

Copyright (c) 2005, 2008, 2012 Andrew Fraser
This file is part of HMM_DS_Code you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later
version.
"""

import sys

def main(argv=None):
    
    import argparse
    import pickle
    import numpy as np
    import ApOb
    if argv is None:                    # Usual case
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=
             '''Study classification performance on training data''')
    parser.add_argument('model',
                        help='A trained HMM for data with classification.')
    parser.add_argument('expert', type=str,
                       help='Path to file of expert annotations')
    parser.add_argument('hr_dir', type=str,
                       help='Path to low pass heart rate data files')
    parser.add_argument('resp_dir', type=str,
                       help='Path to respiration data files')
    parser.add_argument('power', type=float, nargs=3,
              help='from, to, step for study.  Suggest 0.5 2.65 0.1')
    parser.add_argument('fudge', type=float, nargs=3,
              help='from, to, step for study.  Suggest 0.8 1.61 .05')
    parser.add_argument('record', type=str, nargs='*',
                       help='Record names, eg, a01 a02 ... a20')
    args = parser.parse_args(argv)
    model = pickle.load(open(args.model, 'rb'))
    y_mod_0 = model.y_mod.y_mod
    n_seg, segs, data_class = model.y_mod.join(list(ApOb.build_data(
        model.y_mod, args).values()))
    n_y = len(data_class[0])
    data = data_class[1:]
    # data [class, hr, context, resp]
    s2c = model.y_mod.s2c
    for power in np.arange(*args.power):
        for fudge in np.arange(*args.fudge):
            model.y_mod.y_mod = ApOb.fudge_pow(y_mod_0, fudge, power, s2c)
            errors = model.class_decode(data) ^ data_class[0]
            frac_right = 1.0 - errors.sum()/n_y
            print('%5.3f  %5.3f  %6.4f'%(power,fudge,frac_right))
    return 0
        
if __name__ == "__main__":
    argv = [
        'mod_H', 'summary_of_training', 'low_pass_heart_rate',
        'respiration', '3.5', '9.05', '1.0', '0.4', '1.01', '.2'
    ] + list(['a%02d'%x for x in range(1,21)])
    sys.exit(main(argv))

#Local Variables:
#mode:python
#End:

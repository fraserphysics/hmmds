""" ApTrain.py command_options list_of_data_files

EG: python3 ApTrain.py
--hr_dir=../../../derived_data/apnea/low_pass_heart_rate \
--resp_dir=../../../derived_data/apnea/respiration \
--expert=../../../raw_data/apnea/data/summary_of_training \
../../../derived_data/apnea/init_1_1_4 \
../../../derived_data/apnea/modH_1_1_4 \
a06 b01 ...

"""
# Copyright (c) 2005, 2008, 2012 Andrew Fraser
# This file is part of HMM_DS_Code.

# HMM_DS_Code is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.

# HMM_DS_Code is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

import sys
def main(argv=None):
    '''Call with arguments: 

    '''

    if argv is None:                    # Usual case
        argv = sys.argv[1:]

    import argparse
    parser = argparse.ArgumentParser(
        description='Train an existing model on specified data')
    parser.add_argument('--hr_dir', type=str,
                       help='Path to low pass heart rate data files')
    parser.add_argument('--resp_dir', type=str,
                       help='Path to respiration data files')
    parser.add_argument('--expert', type=str,
                       help='Path to file of expert annotations')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of Baum Welch iterations')
    parser.add_argument('mod_in', type=str,
                       help='File from which to read initial model')
    parser.add_argument('mod_out', type=str,
                       help='Write trained model to this file')
    parser.add_argument('record', type=str, nargs='*',
                       help='Record names, eg, a01 a02 ... c09')
    args = parser.parse_args(argv)

    import pickle
    import ApOb
    import Scalar
    mod = pickle.load(args.mod_in)
    y_mod = mod.y_mod
    y_class = y_mod.__class__
    if y_class is Scalar.Class_y:
        readers = [ApOb.read_expert]
        paths = [args.expert]
        args_ = [None]
        y_mod = y_mod.y_mod
        y_class = y_mod.__class__
    else:
        assert args.expert is None
        readers = []
        paths = []
        args_ = []
    if y_class is ApOb.Heart_Rate or y_class is ApOb.Both:
        if y_class is ApOb.Heart_Rate:
            assert args.resp_dir is None
            AR = len(y_mod.A)
        else:
            AR = len(y_mod.hr_mod.A)
        readers.append(ApOb.read_lphr)
        paths.append(args.hr_dir)
        args_.append(AR)
    if y_class is ApOb.Resp or y_class is ApOb.Both:
        if y_class is ApOb.Resp:
            assert args.hr_dir is None
        readers.append(ApOb.read_resp)
        paths.appendargs.resp_dir)
        args_.append(None)
    data_dict = ApOb.read_records(readers, paths, args_, args.record)
    mod.multi_train(data_dict.values(), args.iterations)
    pickle.dump(mod,open(args.mod_out,'w'))
    return 0

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
#Local Variables:
#mode:python
#End:

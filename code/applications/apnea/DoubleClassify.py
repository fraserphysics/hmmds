""" DoubleClassify.py command_options list_of_data_records

For each of the records, first estimate a classification (apnea,
borderline, or normal) for the entire record, then classify (normal or
apnea) each minute in the record.

Or if --Single, just do the first pass.

Copyright (c) 2005, 2008, 2012 Andrew Fraser
This file is part of HMM_DS_Code.


"""
import sys

def main(argv=None):
    
    import argparse
    if argv is None:                    # Usual case
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='''For each named record: 1. Calculate R and llr;
2. Use R and llr to classify entire record as H, M, or L;
3. Classify each minute of the record as normal or apnea.''')
    parser.add_argument('Amodel', help='Used in first pass')
    parser.add_argument('BCmodel', help='Used in first pass')
    parser.add_argument('hr_dir', type=str,
                       help='Path to low pass heart rate data files')
    parser.add_argument('resp_dir', type=str,
                       help='Path to respiration data files')
    parser.add_argument('low_line', type=float,
                       help='Boundary between Low and Medium on first pass')
    parser.add_argument('high_line', type=float,
                       help='Boundary between Medium and High on first pass')
    parser.add_argument('--Single', action='store_true',
              help='Only do the first pass classification')
    parser.add_argument('--report', type=str,
                        help='Location of report to write')
    parser.add_argument('--Lmodel', help='For records that have low score')
    parser.add_argument('--Mmodel', help='For records that have medium score')
    parser.add_argument('--Hmodel', help='For records that have high score')
    parser.add_argument('--expert', type=str,
                       help='Path to file of expert annotations')
    parser.add_argument('record', type=str, nargs='*',
                       help='Record names, eg, a01 a02 ... c09')
    args = parser.parse_args(argv)

    import numpy as np
    import ApOb
    import pickle

    report = open(args.report, 'w')
    Amod = pickle.load(open(args.Amodel, 'rb'))
    BCmod = pickle.load(open(args.BCmodel, 'rb'))
    if not args.Single:
        load = lambda x: pickle.load(open(x, 'rb'))
        Lmod = load(args.Lmodel)
        Mmod = load(args.Mmodel)
        Hmod = load(args.Hmodel)
    data_dict = ApOb.build_data(Amod.y_mod, args) # data [hr, context, resp]
    # Need h_data because AR order for Hmod is different
    if not args.Single:
        h_data = ApOb.build_data(Hmod.y_mod, args, use_class=False)
    for record in args.record:
        data = data_dict[record]
        lp = data[0]              # Scalar low pass heart rate time series
        T = len(lp)
        peaks = []
        L1 = np.abs(lp).sum()/T      # L1 norm of lp per sample
        W = 5                        # Window size for peaks
        #peaks = lp[scipy.signal.argrelmax(lp, order=W)]
        peaks = []
        for t in range(W,T-W):
            s = lp[t-W:t+W].argmax()
            if s == W and lp[t] > 0:
                peaks.append(lp[t])
        peaks.sort()
        R = peaks[int(.74*len(peaks))]/L1
        # Calculate the log likelihood ratio
        Amod.P_Y_calc(data)
        A = Amod.forward()
        BCmod.P_Y_calc(data)
        BC = BCmod.forward()
        llr = (A - BC)/T

        stat = R + .5*llr           # Was 0.5
        if stat < args.low_line:    # Was 2.39
            Name = 'Low'
            if not args.Single:
                model = Lmod
        elif stat > args.high_line: # Was 2.55
            Name = 'High'
            if not args.Single:
                model = Hmod
                data = h_data[record]
        else:
            Name = 'Medium'
            if not args.Single:
                model = Mmod
        print('%3s # %-6s stat= %6.3f llr= %6.3f R= %6.3f'%(
            record, Name, stat, llr, R), end='', file=report)
        if args.Single:
            print('', file=report)
            continue
        Cseq = (model.class_decode(data) - 0.5)*2.0 # +/- 1
        sam_min = 10  # Samples per minute
        min_hour = 60
        sam_hour = 60*sam_min
        for h in range(0, len(Cseq)//sam_hour+1):
            print('\n%-2d   '%h, end='', file=report)
            for m in range(0, 60):
                tot = 0
                for d in range(sam_min):
                    t = d + sam_min * m + sam_hour * h
                    if t >= len(Cseq):
                        break
                    tot += Cseq[t] 
                if t > len(Cseq):
                    break
                if tot > 0:
                    print('A',end='')
                else:
                    print('N',end='', file=report)
        print('\n',end='', file=report)

    return 0
        
if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:

import sys

import numpy


def common_args(parser):
    """ Add arguments required by many functions that operate on the apnea data
    """
    for dir_name in 'heart_rate respiration models data'.split():
        parser.add_argument('--' + dir_name, type=str, help='path to directory')
    parser.add_argument('--iterations',
                        type=int,
                        help='Training iterations',
                        default=20)
    parser.add_argument('--pass1',
                        type=str,
                        help='path to result of pass1',
                        default='pass1_report')


def read_low_pass_heart_rate(path: str) -> numpy.ndarray:
    """Args:
        path: File to read

    Returns:
         array with shape (ntimes,3) and array[i,0] = time in minutes,
         array[i,1] = unfiltered heart rate, array[i,2] = filtered
         heart rate

    Here is the relevant code in hmmds3/code/applications/apnea/rr2hr.py

    HR = rfft(hrL,131072) # 131072 is 18.2 Hrs at 2HZ
    HR[0:100] *=0 # Drop frequencies below (100*60)/65536=0.09/min
    HR[4000:] *=0 # and above (4000*60)/65536=3.66/min
    hrL = irfft(HR)

    """
    with open(path, 'r') as data_file:
        data = [[float(x) for x in line.split()] for line in data_file]
    return numpy.array(data)


def read_respiration(path: str) -> numpy.ndarray:
    """Args:
        path: File to read

    Returns:
         array with shape (ntimes,4) and array[i,0] = time in minutes,
         array[i,1:4] = Respiration vector (?Fisher linear discriminant?)

    The relevant code is hmmds3/code/applications/apnea/respire.py

    """
    with open(path, 'r') as data_file:
        data = [[float(x) for x in line.split()] for line in data_file]
    return numpy.array(data)


if __name__ == "__main__":
    rv = read_low_pass_heart_rate('../../../derived_data/apnea/respiration/a01')
    print(rv[0])
    sys.exit(0)
    #sys.exit(main())

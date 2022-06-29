"""compare.py \
../../../build/derived_data/synthetic/filter/linear_map_data \
../../../build/derived_data/synthetic/filter/mimc_data

Should report that the data in the two pickled dictionaries match.
"""

import sys
import pickle

import numpy


def main(argv=None):
    """Compare two pickle files.
    """
    if argv is None:
        argv = sys.argv[1:]
    data = argv[0]
    mimic_data = argv[1]

    with open(data, 'rb') as _file:
        data = pickle.load(_file)
    with open(mimic_data, 'rb') as _file:
        mimic_data = pickle.load(_file)

    for key in data.keys():
        true = data[key]
        mimic = mimic_data[key]
        if numpy.array_equal(true, mimic):
            print(f'{key} matches')
        else:
            if isinstance(true, numpy.ndarray):
                true = true.flatten()
                mimic = mimic.flatten()
            difference = numpy.abs(true - mimic)
            for i, value in enumerate(difference):
                if value == 0.0:
                    true[i] = 1
            relative = difference / numpy.abs(true)
            print(
                f'{key} differs absolute: {difference.max()} relative: {relative.max()}'
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#---------------
# Local Variables:
# mode:python
# End:

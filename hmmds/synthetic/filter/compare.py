import sys
import pickle

import numpy


def main(argv=None):

    data = pickle.load(open('data', 'rb'))
    mimic_data = pickle.load(open('mimic_data', 'rb'))

    for key in data.keys():
        true = data[key]
        mimic = mimic_data[key]
        if numpy.array_equal(true, mimic):
            print(f'{key} matches')
        else:
            if type(true) == numpy.ndarray:
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

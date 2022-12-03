""" respire.py:  Extract respiration signature from low pass heart rate files

Write a file for plotting that has: The means of the three classes;
The LDA basis vectors; The coefficients of the basis for each minute in
three classes.  Also write a file for each record that gives the
LDA coeeficients for each sample time (tenth of a minute).

Imitates respire.py in my hmmds3 project
"""

import sys
import os.path
import argparse
import pickle
import typing

import pint
import numpy
import numpy.linalg
import scipy.signal

from hmmds.applications.apnea import utilities

PINT = pint.UnitRegistry()


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Calculate respiration from heart rate')
    parser.add_argument('--sample_rate_in',
                        type=int,
                        default=2,
                        help='Samples per second of input')
    parser.add_argument('--sample_rate_out',
                        type=int,
                        default=10,
                        help='Samples per minute for results')
    parser.add_argument('--fft_width',
                        type=int,
                        default=1024,
                        help='Number of samples for each fft')
    parser.add_argument('annotations',
                        type=str,
                        help='File of expert annotations')
    parser.add_argument('heart_rate_dir', type=str, help='Path to heart rate data for reading')
    parser.add_argument('resp_dir', type=str, help='Path to respiration data for writing')
    args = parser.parse_args(argv)
    args.sample_rate_in *=PINT('Hz')
    args.sample_rate_out /= PINT('minutes')
    return args


def spectrogram(filtered_heart_rate, args):
    """Map an array of heart rates sampled at 2 Hz to a spectrogram

    """

    ratio = int((args.sample_rate_in / args.sample_rate_out).magnitude)
    assert ratio == 12
    # With default args ratio is 12
    frequencies, times, psds = scipy.signal.spectrogram(
        filtered_heart_rate.to('Hz').magnitude,
        fs=args.sample_rate_in.to('Hz').magnitude,
        nperseg=args.fft_width,
        noverlap=args.fft_width - ratio,
        detrend=False,
        mode='psd')
    assert len(filtered_heart_rate)/len(times) == ratio
    return frequencies * PINT('Hz'), times * PINT('second'), psds


def linear_discriminant_analysis(# pylint: disable = too-many-locals
        groups: dict,
        records: dict,
        annotation: typing.Callable,
        args) -> dict:
    """Calaculate LDA basis.

    Args:
        groups: EG, groups['a'] = ['a01', 'a02', ..., 'a25']
        records:  Eg, records['a01']['bphr'] = an array of filtered heart rates
        annotation: annotation('a01',1.5) = 0 for time=1.5 minutes.  Or 1 for apnea
        args: Command line arguments

    Returns result, a dict with result['basis'] = 2 lda basis vectors,
    result['c_components'] = 2-d vectors of c records dot lda basis
    result['c_mean'] = mean psd vector and the corresponding data for
    "normal" and "apnea".

    """

    class Class:
        """Collects data and methods for linear discriminant analysis.
        """
        def __init__(self):
            """list will hold psds
            """
            self.list = []

        def mean_covariance(self):
            """Calculate sample mean and covariance within a class.

            """
            # pylint: disable = attribute-defined-outside-init
            self.psds = numpy.array(self.list)
            self.n_samples, self.sample_length = self.psds.shape
            self.mean = numpy.mean(self.psds, axis=0)
            self.covariance = numpy.covariance(self.psds.T)
            assert self.mean.shape == (self.sample_length,)
            assert self.covariance.shape == (self.sample_length,
                                                 self.sample_length)

        def between_term(self, global_mean):
            """Calculate the contribution of this class to between class scatter
            """
            diff = self.mean - global_mean
            return self.n_samples * numpy.outer(diff, diff)

        def components(self, basis):
            """Calculate the components in directions of basis
            """
            return numpy.dot(basis, self.psds)

    # Create arrays of psd vectors for three classes: Normal patients;
    # Apena patients while they are breathing normally; Apena patients
    # while they are experiencing apnea.
    c = Class()  # pylint: disable = invalid-name
    apnea = Class()
    normal = Class()
    for name in groups['c'].values():
        for psd in spectrogram(records[name]['bphr'], args)[-1]:
            c.list.append(psd)
    for name in groups['a'].values():
        _, times, psds = spectrogram(records[name]['bphr'], args)
        for time, psd in zip(times, psds):
            if annotation(name, time):
                apnea.list.append(psd)
            else:
                normal.list.append(psd)
    for _class in (c, apnea, normal):
        _class.mean_covariance()

    global_mean = (c.mean * c.n_samples + apnea.mean * apnea.n_samples +
                   normal.mean * normal.n_samples) / (
                       c.n_samples + apnea.n_samples + normal.n_samples)

    within_class_scatter = c.covariance + apnea.covariance + normal.covariance

    between_class_scatter = c.between_term(global_mean) + apnea.between_term(
        global_mean) + normal.between_term(global_mean)

    within_information = numpy.linalg.inv(
        within_class_scatter + numpy.eye(len(within_class_scatter)))
    values, vectors = numpy.linalg.eigh(
        numpy.dot(within_information, between_class_scatter))
    big2 = numpy.argsort(values)[-2:]
    basis = vectors[:, big2]
    result = {
        'basis': basis,
        'c_mean': c.mean,
        'apnea_mean': apnea.mean,
        'normal_mean': normal.mean,
        'c_components': c.components(basis),
        'apnea_components': apnea.components(basis),
        'normal_components': normal.components(basis)
    }
    return result


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    # Read heart rate data
    annotations = {}
    records = {}
    groups = {'a': [], 'b': [], 'c': [], 'x': []}
    for name in os.listdir(args.lphr_dir):
        assert name[0] in 'abcx'
        assert int(name[-2:]) > 0  # Ensure that name ends in digits
        # assign a01 data to records['a01']
        with open(os.path.join(args.lphr_dir, name + '.lphr'), 'wb') as _file:
            records[name] = pickle.load(_file)
        groups[name[0]].append(name)
        assert records[name]['sample_frequency'] == args.sample_rate_in*PINT('Hz')
        if name[0] != 'x':
            annotations[name] = utilities.read_expert(args.annotations, name)


    # Do linear discriminant analysis and write summary
    def annotation(name, time):
        """annotation('a01', 15.7) -> 0 or 1, 0 for normal
        """
        i_time = int(time.to('minutes').magnitude)  # int(.9) = 0
        return annotations[name][i_time]
    result = linear_discriminant_analysis(groups, records, annotation, args)
    with open(os.path.join(args.resp_dir, 'lda_data'), 'wb') as _file:
        pickle.dump(result, _file)

    # Write lda components for each record
    basis = result['basis']
    for name, record in records.items():
        frequencies, times, psds = spectrogram(record['bphr'], args)
        assert psds.shape == (len(times), len(frequencies))
        assert basis.shape == (2, len(frequencies))
        components = numpy.dot(basis, psds)
        assert isinstance(components, numpy.ndarray)
        assert components.shape == (len(times), 2)
        with open(os.path.join(args.resp_dir, name+'.resp'), 'wb'):
            pickle.dump((times,components))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:

"""lda.py: Make two figures to illustrate the linear discriminant analysis

"""

import sys
import argparse
import pickle
import os

#import pint
import numpy

import plotscripts.utilities

#PINT = pint.UnitRegistry()
#pint.set_application_registry(PINT)  # Makes objects from pickle.load
# use this pint registry.  FixMe: Put the Nyquist frequency in the lda_dict

def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(
        description='Two figures to illustrate linear discriminant analysis')
    parser.add_argument('--apnea_data_dir', type=str, default='../../../build/derived_data/apnea')
    # respire_subdir and lda_data are options to avoid having magic
    # constants in the code.
    parser.add_argument('--respire_subdir', type=str, default='Respire')
    parser.add_argument(
        '--lda_name', type=str, default='lda_data', help='Name of file that has lda data')
    parser.add_argument('--show',
                        action='store_true',
                        help='display figure in pop-up window')
    parser.add_argument('figure_1', type=str, help='Path to one of two results')
    parser.add_argument('figure_2', type=str, help='Path to one of two results')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    """Make linear discriminant analysis figures
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    args, matplotlib, pyplot = plotscripts.utilities.import_and_parse(
        parse_args, argv)

    respire_dir = os.path.join(args.apnea_data_dir, args.respire_subdir)
    with open(os.path.join(respire_dir, args.lda_name), 'rb') as _file:
        lda_dict = pickle.load(_file)

    # Check and document items in lda_dict.  FixMe: Should the
    # normalization factor be part of the LDA?
    two, length_raw = lda_dict['basis'].shape
    assert two == 2
    for _class in 'c apnea normal'.split():
        assert lda_dict[_class+'_mean'].shape == (length_raw,)
        assert lda_dict[_class+'_components'].shape[0] == 2

    frequencies = 120*numpy.arange(length_raw)/length_raw # FixMe: Explain 120
    figure1, (ax_means, ax_components) = pyplot.subplots(
        nrows=2, ncols=1, figsize=(5, 5.5), sharex=True)

    n_limit = int(length_raw*.3)
    def limited_plot(ax, x, y, n_limit, color, label):
        ax.plot(x[:n_limit], y[:n_limit], color, label=label)
        
    for _class, color, label in zip(
            # _class
            'c_mean apnea_mean normal_mean'.split(),
            # color
            'r- g- b-'.split(),
            # label
            [r'$\mu_%s$'%(c,) for c in'C A N'.split()]):
        limited_plot(ax_means, frequencies, lda_dict[_class], n_limit, color, label)

    for i, color, label in zip(
            # Component
            (0,1),
            # Color
            'r- b-'.split(),
            # Label
            ('1','2')):
        limited_plot(ax_components, frequencies, lda_dict['basis'][i], n_limit, color, f'$v_{label}$')

    for ax in (ax_means, ax_components):
        ax.set_yticks([])
        ax.set_ylabel(r'PSD')
        ax.legend(loc='upper right')
    ax_components.set_xlabel(r'cpm FixMe') #FixMe
    
    figure2, (ax_c, ax_n, ax_a, ax_all) = pyplot.subplots(
        nrows=4, ncols=1, figsize=(6, 15))
    box = [-0.6, 0.65, -0.4, 0.9]
    for (ax, _class, color, label) in (
            (ax_c, 'c', 'r', 'C'),
            (ax_a, 'apnea', 'g', 'A'),
            (ax_n, 'normal', 'b', 'N'),
            ):
        x,y = lda_dict[_class+'_components']
        # FixMe: Plot one sample per minute because expert classified
        # by minute
        ax.plot(x[3::6],y[3::6],color, marker=',', markersize=1, linestyle='None')
        ax_all.plot(x[3::6],y[3::6],color, marker=',', markersize=1, linestyle='None')
        ax.axis(box)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend((label,), loc='lower right')
    ax_all.axis(box)

    figure1.savefig(args.figure_1)
    figure2.savefig(args.figure_2)
    if args.show:
        pyplot.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""particle.py Apply variant of particle filter to quantized Lorenz data

Call with: python study_particle.py result_file

Create x_all, y_q and p_filter (initialized with x_all[0]).  Then run
filter on y_q.

Write a dict to a pickle file with the following keys:

x_all  Long (args.n_initialize) Lorenz trajectory
y_q    Quantized data derived from x_all[:args.n_y]
gamma  Incremental likelihood from p_filter(y_q, ...)
clouds Forecast and update particles at times args.clouds
bins   Bin boundaries.

"""

from __future__ import annotations
import sys
import pickle

import numpy
import numpy.linalg

from hmmds.synthetic.bounds import lorenz
from hmmds.synthetic.bounds import benettin
from hmmds.synthetic.bounds.filter import Filter
from hmmds.synthetic.bounds.particle import parse_args, make_data, initialize

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as pyplot


def plot(gammas):
    ''''''

    figure, axeses = pyplot.subplots(nrows=2, ncols=2, figsize=(8, 9))
    offset = 14
    for key, gamma in gammas.items():
        log_gamma = numpy.log(gamma)[offset:]
        cum_sum = numpy.cumsum(log_gamma)
        y_values = -cum_sum / numpy.arange(1, len(cum_sum) + 1) / 0.15
        axeses[1, 0].plot(numpy.log10(gamma))
        axeses[0, 1].plot(numpy.arange(offset, offset + len(y_values)),
                          y_values,
                          label=rf'$\hat h$ {key}')
    axeses[0, 1].legend()
    pyplot.show()


def run_filter(args, y_all, x_all, bins):
    ''''''
    rng = numpy.random.default_rng(args.random_seed)
    p_filter = Filter(args, bins, rng)
    initialize(p_filter, y_all, y_all, x_all[args.n_y:], x_all[0])
    gamma = numpy.ones(len(y_all))
    print(f'''{y_all.shape=} {x_all.shape=} {y_all.shape=}
    ''')
    # Run filter on y_all
    p_filter.forward(y_all, (0, len(y_all)), gamma)
    return gamma


def main(argv=None):
    """Run particle filter on Lorenz data

    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    y_all, x_all, bins = make_data(args)

    gammas = {}
    # margin    h       n_y 1000
    # 2.0       1.0549
    #  .6       1.0606
    #  .5       1.0128
    #  .4       1.03272
    #  .1       1.0441
    #  .01      1.02237
    # .006      1.03    dies at 600
    # .003      1.0047
    # .001      1.07    dies at 281
    args.margin = 0.01  # Default .5
    # s_augment h       n_y 1000 margin 0.01
    # 1e-1      1.233
    # 1e-2      1.080
    # 5e-3      1.0498
    # 5e-3      1.015   at 400
    # 1e-3      1.020   ***
    # 5e-4      1.0227
    # 5e-4      0.9806  at 400
    # 1e-4      1.039
    # 1e-5      1.056
    # 1e-6      1.069
    ##### with resample = (100000, 40000)
    # 1e-3      1.004
    # 5e-4      0.987
    # 1e-4      1.020
    # 5e-5      1.025
    # 1e-5      1.009
    # 5e-6      1.033
    args.s_augment = 5.0e-4  # Default 5.0e-4
    # resample     h at 400
    # 5000  2000   1.0279
    # 10000 4000   0.9806
    # 10000 2000   dies 174
    # 20000 4000   dies 146
    # 20000 2000   dies at 264
    # 20000 8000   0.9807
    # 40000 5000   1.0039
    # 40000 10000  0.9914
    # 100000 5000  1.0300
    # 100000 10000 1.0319
    # 100000 20000 1.0123
    # 100000 30000 1.0039
    # 100000 40000 0.9866
    # 100000 50000 1.0133
    args.resample = (100000, 40000)  # Default (10000, 4000)
    # r_threshold  h at 400
    # 1e-3      0.9865
    # 5e-4      1.0114
    # 1e-4      0.9810
    # 5e-5      0.9365
    # 1e-5      dies at 265   Has low entropy
    # 5e-6      dies at 144.  Has lowest entropy  ********* Try to keep this from dying.
    args.r_threshold = 5e-5  # Default 0.001
    # r_extra Default 2
    # r_extra   h at 400
    # 1         0.9381
    # 2         0.9365
    # 4         dies at 164
    # 8         dies at 146
    # edge_max  h at 400
    # .1        .9223
    # .2        .9365
    # .3        .9366
    # edge_max Default .2
    for edge_max in (.1, .3):
        args.edge_max = edge_max
        print(f'try {edge_max=}')
        gammas[f'{edge_max=}'] = run_filter(args, y_all, x_all, bins)
    plot(gammas)
    # Write results
    result = {'gamma': gammas}

    with open(args.result, 'wb') as _file:
        pickle.dump(result, _file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Local Variables:
# mode: python
# End:

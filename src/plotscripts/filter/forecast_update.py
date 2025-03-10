"""ddays_plot_b.py Makes figure for dynamics days 2025

python ddays_plot_b.py input_path output_path

input is made by hmmds/synthetic/bounds/particle.py

"""

import sys
import argparse
import pickle

import numpy
import numpy.linalg

import plotscripts.utilities


def parse_args(argv):
    """ Convert command line arguments into a namespace
    """

    parser = argparse.ArgumentParser(description='Debugging plot')
    parser.add_argument('--show',
                        action='store_true',
                        help="display figure using Qt5")
    parser.add_argument('--start',
                        type=int,
                        default=72,
                        help='Plot particles at 2 times starting here')
    parser.add_argument('--t_rows',
                        type=int,
                        nargs='*',
                        default=(0, 10, 31),
                        help='time for first plot in each row')
    parser.add_argument('input', type=str, help='Path to data')
    parser.add_argument('--no_divide', type=str, help='Path to figure file')
    parser.add_argument('--with_divide', type=str, help='Path to figure file')
    parser.add_argument('--entropy', type=str, help='Path to figure file')
    args = parser.parse_args(argv)
    return args


def plot_point(axes, x, color, label=None):
    axes.plot(x[0],
              x[2],
              markeredgecolor='none',
              marker='.',
              markersize=5,
              linestyle='None',
              color=color,
              label=label)


def plot_selected(axes, x_all, bins, indices: set, shift, label=None):
    """Plot points selected by indices shifted by steps

    Args:
        axes: Plot on this
        x_all: Long vector time series
        bins: For vertical lines
        indices: Unshifted times
        shift: Plots points at times = indices + shift
    """
    index_array = numpy.asarray([index for index in indices])
    shifted_indices = (index_array + shift,)
    axes.plot(
        x_all[shifted_indices, 0],
        x_all[shifted_indices, 2],
        markeredgecolor='none',
        color='#1f77b4',
        marker='.',
        markersize=2.5,
        linestyle='None',
    )
    axes.plot(x_all[shift, 0],
              x_all[shift, 2],
              marker='x',
              markersize=5,
              linestyle='None',
              color='red',
              label=label)
    for boundary in bins:
        axes.plot((boundary,) * 2, (0, 50), color='black', linewidth=.5)
    axes.set_xlim(-22, 22)
    axes.set_ylim(0, 50)
    if label:
        axes.legend()


def no_divide_figure(t_rows, x_all, bins, pyplot):
    """Plots as number of points in initial cloud decays to zero

    """
    y_all = numpy.digitize(x_all[:, 0], bins)
    n_cols = 4
    last_time = t_rows[-1] + n_cols
    plotable = {}
    for row, t_start in enumerate(t_rows):
        for column in range(n_cols):
            plotable[t_start + column] = (row, column)

    figure, axeses = pyplot.subplots(nrows=2 * len(t_rows),
                                     ncols=n_cols,
                                     figsize=(8, 12),
                                     sharex=True,
                                     sharey=True)

    update_indices = set(numpy.arange(1000, len(x_all) - last_time))

    for shift in range(last_time):
        forecast_indices = update_indices
        update_indices = forecast_indices & set(
            numpy.nonzero(y_all[shift:] == y_all[shift])[0])
        if shift not in plotable:
            continue
        row, column = plotable[shift]
        forecast_axes = axeses[row * 2, column]
        update_axes = axeses[row * 2 + 1, column]
        n_forecast = len(forecast_indices)
        n_update = len(update_indices)
        if n_forecast > 0:
            p = n_update / n_forecast
        else:
            p = 0
        plot_selected(forecast_axes,
                      x_all,
                      bins,
                      forecast_indices,
                      shift,
                      label=f'n[{shift}]={n_forecast}')
        plot_selected(update_axes,
                      x_all,
                      bins,
                      update_indices,
                      shift,
                      label=f'P({shift})={p:.2f}')

        if column != 0:
            continue
        forecast_axes.set_ylabel(r'$\rm{Forecast}$')
        update_axes.set_ylabel(r'$\rm{Update}$')
        forecast_axes.set_yticks([])
        update_axes.set_yticks([])
    return figure


def with_divide_figure(t_rows, clouds, x_all, bins, pyplot):
    """Plots of clouds of particles

    """

    n_cols = 4
    plotable = {}
    last_time = t_rows[-1] + n_cols
    for row, t_start in enumerate(t_rows):
        for column in range(n_cols):
            plotable[t_start + column] = (row, column)

    figure, axeses = pyplot.subplots(nrows=2 * len(t_rows),
                                     ncols=n_cols,
                                     figsize=(8, 12),
                                     sharex=True,
                                     sharey=True)
    for i in range(t_rows[0], last_time):
        if i not in plotable:
            continue
        row, column = plotable[i]
        forecast_axes = axeses[row * 2, column]
        update_axes = axeses[row * 2 + 1, column]
        forecast = clouds[(i, 'forecast')]
        update = clouds[(i, 'update')]

        # Plot points of forecast and update
        for cloud, axes in ((forecast, forecast_axes), (update, update_axes)):
            for particle in cloud:
                plot_point(axes, particle.x, '#1f77b4')
            for boundary in bins:
                axes.plot((boundary,) * 2, (0, 50), color='black', linewidth=.5)
            axes.plot(
                x_all[i, 0],
                x_all[i, 2],
                marker='x',
                markersize=5,
                linestyle='None',
                color='red',
            )
            axes.set_xlim(-22, 22)
            axes.set_ylim(0, 50)
        plot_point(forecast_axes, forecast[0].x, '#1f77b4',
                   f'n[{i}]={len(forecast)}')
        p = len(update) / len(forecast)
        plot_point(update_axes, update[0].x, '#1f77b4', f'p[{i}]={p:.2f}')
        forecast_axes.legend()
        update_axes.legend()
        if column != 0:
            continue
        forecast_axes.set_ylabel(r'$\rm{Forecast}$')
        update_axes.set_ylabel(r'$\rm{Update}$')
    return figure


def entropy(gamma, pyplot):
    """"""

    offset = 14
    log_gamma = numpy.log(gamma)[offset:]
    cum_sum = numpy.cumsum(log_gamma)
    y_values = -cum_sum / numpy.arange(1, len(cum_sum) + 1) / 0.15

    figure, axes = pyplot.subplots(figsize=(6, 4))

    axes.plot(numpy.arange(offset, offset + len(y_values)),
              y_values,
              label=r'$\hat h$')
    x_max = len(y_values)
    y_level = 0.906
    axes.plot([0, x_max], [y_level, y_level], label=r'$\lambda$')
    axes.set_ylabel(r'$\hat h/\rm{nats}$')
    axes.set_xlabel(r'$n_{\rm{samples}}$')

    axes.set_ylim(0, 2.0)
    min_y, max_y = axes.get_ylim()
    min_x, max_x = axes.get_xlim()
    ax2 = axes.twinx()
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)
    ax2.set_yticks((
        .906,
        y_values[-1],
    ))
    axes.legend()
    return figure


def main(argv=None):
    """Plot some stuff
    """

    args, _, pyplot = plotscripts.utilities.import_and_parse(parse_args, argv)

    with open(args.input, 'rb') as file_:
        dict_in = pickle.load(file_)
    gamma = dict_in['gamma']  # (100,)
    y_q = dict_in['y_q']  # (100,)  Starts at beginning of x_all
    bins = dict_in['bins']  # (3,)
    x_all = dict_in['x_all']  # (15100, 3)
    clouds = dict_in['clouds']  # dict
    if args.no_divide:
        figure = no_divide_figure(args.t_rows, x_all, bins, pyplot)
        figure.tight_layout()
        if args.show:
            pyplot.show()
        else:
            figure.savefig(args.no_divide)

    if args.with_divide:
        figure = with_divide_figure(args.t_rows, clouds, x_all, bins, pyplot)
        figure.tight_layout()
        if args.show:
            pyplot.show()
        else:
            figure.savefig(args.with_divide)

    if args.entropy:
        figure = entropy(gamma, pyplot)
        figure.tight_layout()
        if args.show:
            pyplot.show()
        else:
            figure.savefig(args.entropy)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""tex_values.py Create a file for moving values from code to tex

"""

from __future__ import annotations  # Enables, eg, (self: HMM,

import sys
import argparse

import lorenz


def parse_args(argv):
    """ Combine command line arguments with defaults from utilities
    """

    parser = argparse.ArgumentParser(
        "Create a file for moving values from code to tex")
    parser.add_argument('sbr',
                        type=float,
                        nargs=3,
                        help='Parameters of Lorenz system')
    parser.add_argument('levels', type=int)
    parser.add_argument("command_line",
                        type=str,
                        nargs='+',
                        help="Get key value pairs from command line")
    parser.add_argument('write_path', type=str, help='path of file to write')
    args = parser.parse_args(argv)
    return args


def command_line(args, result: dict):
    assert len(args.command_line) % 2 == 0
    for key, value in zip(args.command_line[0::2], args.command_line[1::2]):
        result[key] = value


def main(argv=None):
    """Create a file for moving values from code to tes
    """
    if argv is None:  # Usual case
        argv = sys.argv[1:]

    args = parse_args(argv)

    result = dict(
        ((key, value) for key, value in zip('S B R'.split(), args.sbr)))

    for (key, level) in zip('boundaryA boundaryB boundaryC'.split(),
                            lorenz.get_bins(args)):
        result[key] = level

    command_line(args, result)

    with open(args.write_path, 'w', encoding='utf-8') as file_:
        for key, value in result.items():
            file_.write(f'\def\synthetic{key}{{{value}}}\n')

    return 0


if __name__ == "__main__":
    sys.exit(main())

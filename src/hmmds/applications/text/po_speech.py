''' po_speech.py: For discovering parts of speech for first chapter of book

Use: python po_speech.py input_book_path out_file_path

'''

import sys
import argparse
import re

import numpy
import numpy.random

import hmm.simple
import hmm.C


def parse_args(argv):
    """Parse the command line.
    """
    parser = argparse.ArgumentParser(
        description='Train an HMM on words in a book and decode states')
    parser.add_argument('--n_iterations', type=int, default=200)
    parser.add_argument('--n_states', type=int, default=15)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('in_path', type=str)
    parser.add_argument('latex_values_path',
                        type=str,
                        help='Write a file of \defs to this path')
    parser.add_argument('table_path',
                        type=str,
                        help='Write a LaTeX table to this path')
    return parser.parse_args(argv)


def read_text(text) -> list:
    '''Read from "text" and return a sequence of words or "tokens".

    Args:
        text: A file open for reading

    Return: A list of tokens

    This code removes the first 334 tokens and the last 5,864 tokens
    of the Project Gutenberg EBook of "A Book of Prefaces", by
    H. L. Mencken.  That's a mix of stuff that project gutenberg added
    and the index of the book.

    '''
    all_ = re.sub('-\n', '', text.read())
    all_ = re.sub('--+', '--', all_)
    all_ = re.sub('\.\.\.+', '\.\.\.', all_)
    # The following sub-patterns match ordinary words, punctuation, times,
    # punctuated numbers and unpuncuated integers.
    pattern = "[a-zA-Z']+|"                   \
              +'["'+":;,!.?()$*`'\[\]<>&/-]|" \
              +"[0-9]+:[0-9]+|"               \
              +"[0-9]+[0-9.,]*[0-9]+|"        \
              +"[0-9]+"
    token_sequence = re.findall(pattern, all_)[334:-5864]
    return token_sequence


def make_token2int(token_sequence: list) -> tuple:
    r'''Sort tokens in token_sequence by frequency and map tokens to integers

    Args:
        token_sequence: A list of tokens

    Return: (token2int, cardinality_Y, token_list)

    token2int: token2int['the']=1 because 'the' is second most frequent token.
    cardinality_Y: Observations, y[t] \in [0:cardinality_Y]
    token_list: List of pairs (token, max(occurances,2)) sorted by occurances

    Infrequent tokens are all mapped to the same integer.
    '''
    token2int = {}
    for token in token_sequence:
        if token in token2int:
            token2int[token] += 1
        else:
            token2int[token] = 1
    # Now "token2int" is a dict of all tokens that occur.  Each key is a token
    # and the value is the number of occurrences

    token_list = list(token2int.items())
    token_list.sort(key=lambda x: -x[1])
    # Now "token_list" is list of tuples (token,n_occurrence) sorted by
    # n_occurrence

    # Identify "merge; the beginning of the tail of "token_list" where
    # occurrence is <= bottom.  All tokens that occur <= bottom often
    # get mapped to the same integer value of y.
    bottom = 2
    for n in range(len(token_list)):
        key, count = token_list[n]
        if count <= bottom:
            merge = n
            break
    token_list[merge] = ('****', bottom)

    # Change the value of each entry in the dict "token2int" to be
    # minimum of the token rank and "merge"
    for index, (key, count) in enumerate(token_list):
        if count > bottom:
            token2int[key] = index
        else:
            token2int[key] = merge
    return token2int, merge + 1, token_list


def random_hmm(cardinality_Y, n_states, seed):
    """Create and return a hmm.C.HMM

    Args:
       cardinality_Y: Possible values of y is [0:cardinality_Y]
       n_states: Possible values of state is [0:n_states]
       seed: For random number generator
    """
    rng = numpy.random.default_rng(seed)

    def random_prob(shape):
        return hmm.simple.Prob(rng.random(shape)).normalize()

    p_s0 = random_prob((1, n_states))[0]
    p_s0_ergodic = random_prob((1, n_states))[0]
    p_s_to_s = random_prob((n_states, n_states))
    p_s_to_y = random_prob((n_states, cardinality_Y))
    observation_model = hmm.simple.Observation(p_s_to_y, rng)
    return hmm.C.HMM(p_s0, p_s0_ergodic, p_s_to_s, observation_model, rng)


def write_latex(args, cardinality_Y: int, token_list: list,
                y_sequence: numpy.ndarray, state_sequence: numpy.ndarray):
    '''Print the most frequent 10 tokens associated with each state

    Args:
        args: Command line arguments
        cardinality_Y: Number of different values of y[t]
        token_list: List of pairs (token, max(frequency,2)) sorted by frequency
        y: Sequence of integer observations
        state_sequence: Sequence of integer indices of states
    '''
    file_ = open(args.table_path, 'w', encoding='utf-8')

    print(r"""\begin{tabular}{
|@{\hspace{0.10em}}r@{\hspace{0.40em}}|
*{10}{@{\hspace{0.28em}}l@{\hspace{0.28em}}}
|}
\hline""",
          file=file_)
    for n_state in range(args.n_states):
        token_counter = [[y_n, 0] for y_n in range(cardinality_Y)]
        for t, state_t in enumerate(state_sequence):
            if state_t != n_state:
                continue
            if y_sequence[t] == cardinality_Y - 1:  # Ignore infrequent tokens
                continue
            token_counter[int(y_sequence[t])][1] += 1
        token_counter.sort(key=lambda x: -x[1])
        print(f'{n_state+1:2d}', end=' ', file=file_)
        for i in range(10):
            token = token_list[token_counter[i][0]][0]
            if token == '&':
                token = '\&'
            print(f'& {token} ', end=' ', file=file_)
        print(r'\\', file=file_)
    print(r"""\hline
\end{tabular}""", file=file_)
    file_.close()


def write_values(args, value_dict):
    """Write file of \defs for \input in LaTeX file
    """
    with open(args.latex_values_path, 'w', encoding='utf-8') as file_:
        for key, value in value_dict.items():
            print(fr'\def\text{key}{{{value}}}', file=file_)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    with open(args.in_path, 'r', encoding='utf-8') as file_:
        token_sequence = read_text(file_)
    token2int, cardinality_Y, token_list = make_token2int(token_sequence)
    # Map tokens in "token_sequence" to integers in Y
    y = numpy.empty(len(token_sequence), numpy.int32)
    for t, token in enumerate(token_sequence):
        y[t] = token2int[token]
    model = random_hmm(cardinality_Y, args.n_states, args.random_seed)
    model.train(y, args.n_iterations, display=False)
    # Do Viterbi decoding
    state_sequence = model.decode(y)
    assert state_sequence.shape == (len(token_sequence),)
    write_latex(args, cardinality_Y, token_list, y, state_sequence)
    write_values(
        args, {
            'MoreThanTwicePlusOne': cardinality_Y,
            'MoreThanTwice': cardinality_Y - 1,
            'NTokens': len(token_sequence),
            'NUniqueTokens': len(token2int),
            'TrainingIterations': args.n_iterations,
            'NStates': args.n_states
        })


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:

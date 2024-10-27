''' po_speech.py input_book_path out_file_path
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
    parser.add_argument('--n_iterations', type=int, default=100)
    parser.add_argument('--n_states', type=int, default=15)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str)
    return parser.parse_args(argv)


def read_text(text):
    ''' read from "text" and return sequence of words
    '''
    all = re.sub('-\n', '', text.read())
    all = re.sub('--+', '--', all)
    all = re.sub('\.\.\.+', '\.\.\.', all)
    # The following sub-patterns match ordinary words, punctuation, times,
    # punctuated numbers and unpuncuated integers.
    pattern = "[a-zA-Z']+|"                   \
              +'["'+":;,!.?()$*`'\[\]<>&/-]|" \
              +"[0-9]+:[0-9]+|"               \
              +"[0-9]+[0-9.,]*[0-9]+|"        \
              +"[0-9]+"
    all = re.findall(pattern, all)
    return all


def all2words(all):
    '''
    '''
    words = {}
    for word in all:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    # Now "words" is a dict of all words that occur.  Each key is a word
    # and the value is the number of occurrences
    word_list = list(words.items())
    word_list.sort(key=lambda x: -x[1])
    # Now "word_list" is list of tuples (word,occurrence) sorted by
    # occurrence

    # Identify "merge; the beginning of the tail of "word_list" where
    # occurrence is <= 2
    bottom = 2
    for n in range(len(word_list)):
        key, count = word_list[n]
        if count <= bottom:
            merge = n
            break
    word_list[merge] = ('****', bottom)
    # Change value of each entry in dict "words" to be minimum of the word
    # rank and "merge"
    for n in range(len(word_list)):
        key, count = word_list[n]
        if count > bottom:
            words[key] = n
        else:
            words[key] = merge
    return words, merge, word_list


def random_hmm(n_y, n_states, seed):
    """Create and return a hmm.C.HMM
    """
    rng = numpy.random.default_rng(seed)

    def random_prob(shape):
        return hmm.simple.Prob(rng.random(shape)).normalize()

    p_s0 = random_prob((1, n_states))[0]
    p_s0_ergodic = random_prob((1, n_states))[0]
    p_s_to_s = random_prob((n_states, n_states))
    p_s_to_y = random_prob((n_states, n_y))
    observation_model = hmm.simple.Observation(p_s_to_y, rng)
    return hmm.C.HMM(p_s0, p_s0_ergodic, p_s_to_s, observation_model, rng)


def write_latex(args, merge, word_list, y, ss):
    # Print the most frequent 10 words associated with each state
    f = open(args.out_path, 'w')
    print(r"""\begin{tabular}{
|@{\hspace{0.10em}}r@{\hspace{0.40em}}|
*{10}{@{\hspace{0.28em}}l@{\hspace{0.28em}}}
|}
\hline""",
          file=f)
    for s_n in range(args.n_states):
        s_words = list(range(merge))
        for n in range(merge):
            s_words[n] = [n, 0]
        for t in range(len(ss)):
            if ss[t] != s_n:
                continue
            if y[t] == merge:
                continue
            s_words[int(y[t])][1] += 1
        s_words.sort(key=lambda x: -x[1])
        print('%d' % (s_n + 1), end=' ', file=f)
        for i in range(10):
            print('&%s' % word_list[s_words[i][0]][0], end=' ', file=f)
        print(r'\\', file=f)
    print("""\hline
\end{tabular}""", file=f)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    all = read_text(open(args.in_path, 'r', encoding='utf-8'))
    words, merge, word_list = all2words(all)
    # Map words in "all" to integers in "y"
    y = numpy.empty(len(all), numpy.int32)
    for n in range(len(all)):
        y[n] = words[all[n]]
    Card_Y = merge + 1
    model = random_hmm(Card_Y, args.n_states, args.random_seed)
    print("""
Begin training in po_speech.py.  100 iterations in 8 seconds on an AMD Ryzen 9 7950X.
""",
          file=sys.stderr)
    LL = model.train(y, args.n_iterations, display=False)
    # Do Viterbi decoding
    ss = model.decode(y)
    write_latex(args, merge, word_list, y, ss)


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:

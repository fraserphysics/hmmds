''' po_speech.py book_path out_file
'''

import numpy, sys, Scalar, random, re

N_states=15
iterations=100
random.seed(3)
def read_text(text):
    ''' read from "text" and return sequence of words
    '''
    all = re.sub('-\n','',text.read())
    all = re.sub('--+','--',all)
    all = re.sub('\.\.\.+','\.\.\.',all)
    # The following sub-patterns match ordinary words, punctuation, times,
    # punctuated numbers and unpuncuated integers.
    pattern = "[a-zA-Z']+|"                   \
              +'["'+":;,!.?()$*`'\[\]<>&/-]|" \
              +"[0-9]+:[0-9]+|"               \
              +"[0-9]+[0-9.,]*[0-9]+|"        \
              +"[0-9]+"
    all = re.findall(pattern,all)
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
        key,count = word_list[n]
        if count <= bottom:
            merge = n
            print('%%bottom=%d merge=%d len(word_list)=%d'%(
                bottom,merge,len(word_list)))
            break
    word_list[merge] = ('****',bottom)
    # Change value of each entry in dict "words" to be minimum of the word
    # rank and "merge"
    for n in range(len(word_list)):
        key,count = word_list[n]
        if count > bottom:
            words[key] = n
        else:
            words[key] = merge
    return words, merge, word_list

def random_hmm(Card_Y, N_states):
    '''
    '''

    from Scalar import make_random as random
    import base
    P_S0 = random((1,N_states))[0]
    P_S0_ergodic = random((1,N_states))[0]
    P_ScS = random((N_states,N_states))
    P_YcS = random((N_states,Card_Y))
    return base.HMM(P_S0, P_S0_ergodic, P_YcS, P_ScS)

def main(argv=None):
    if argv is None:                    # Usual case
        argv = sys.argv[1:]
    assert len(argv) == 2
    in_name, result_name = argv

    all = read_text(open(in_name,'r'))
    words, merge, word_list = all2words(all)
    # Map words in "all" to integers in "y"
    y = numpy.empty(len(all),numpy.int32)
    for n in range(len(all)):
        y[n] = words[all[n]]
    Card_Y = merge + 1
    model = random_hmm(Card_Y, N_states)
    print("""
Begin training in po_speech.py.  Takes 26 minutes on a 1 GHZ 64bit Athlon.
""", file=sys.stderr)
    LL = model.train([y], iterations,display=False)
    # Do Viterbi decoding
    ss = model.decode([y])
    # Print the most frequent 10 words associated with each state
    f = open(result_name, 'w')
    
    for s_n in range(N_states):
        s_words = list(range(merge))
        for n in range(merge):
            s_words[n] = [n,0]
        for t in range(len(ss)):
            if ss[t] != s_n:
                continue
            if y[t] == merge:
                continue
            s_words[int(y[t])][1] += 1
        s_words.sort(key=lambda x: -x[1])
        print('\\rule{0pt}{2.0ex} %d'%(s_n+1), end=' ', file=f)
        for i in range(10):
            print('&%s'%word_list[s_words[i][0]][0], end=' ', file=f)
        print('\\\\', file=f)


if __name__ == "__main__":
    sys.exit(main())

#Local Variables:
#mode:python
#End:

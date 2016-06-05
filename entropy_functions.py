import numpy as np
from collections import defaultdict

def unigram_prob(unigram_dict, total_words):
  '''
  from a list of unigrams with the count of their ocurrences
  it returns the probability of each unigram
  '''
  probs = {}
  for word in unigram_dict:
    probability = unigram_dict[word] / total_words
    probs[word] = probability

  return probs


def bigram_prob(bigram_dict, unigrams):
  '''
  from a list of bigrams with the count of their ocurrences
  it returns the probability of each bigram and a dictionary with
  the posible combinations for each word, e.g. { 'the': ['pets', 'dogs'] }
  '''
  probs = {}
  b_posibilities = defaultdict(list)
  for bigram in bigram_dict:
    b_posibilities[bigram[0]].append(bigram[1])
    probability = bigram_dict[bigram] / float(unigrams[bigram[0]])
    probs[bigram] = probability

  return probs, b_posibilities


def trigram_prob(trigram_dict, bigrams):
  '''
  from a list of trigrams with the count of their ocurrences
  it returns the probability of each trigram and a dictionary with
  the posible combinations for each word, e.g. { ('I', 'am'): ['Barbara', 'tall', 'Mexican'] }
  '''
  probs = {}
  t_posibilities = defaultdict(list)
  for trigram in trigram_dict:
    t_posibilities[trigram[0], trigram[1]].append(trigram[2])
    probability = trigram_dict[trigram] / float(bigrams[trigram[0],trigram[1]])
    probs[trigram] = probability

  return probs, t_posibilities


def unigram_model(words, probs):
  '''
  function that returns the unigram entropy
  '''
  h = 0.0
  for word in words:
    p = probs[word]
    h += p * np.log2(p)

  return (-1 * h)


def bigram_model(unigrams, b_posibilities, u_probs, b_probs):
  '''
  function that returns the bigram entropy using the unigram and bigram probabilities.
  '''
  h = 0.0
  for unigram in unigrams:
    pb = 0.0
    posibilities = b_posibilities[unigram]
    for pos in posibilities:
      p = b_probs[(unigram, pos)]
      pb += p * np.log2(p)
    h += pb * u_probs[unigram]

  return (-1 * h)


def trigram_model(unigrams, b_posibilities, t_posibilities, u_probs, b_probs, t_probs):
  '''
  function that returns the trigram entropy using the unigram, bigram and trigram probabilities.
  '''
  h = 0.0
  for unigram in unigrams:
    pb = 0.0
    b_pos = b_posibilities[unigram]
    for b_p in b_pos:
      pt = 0.0
      p = b_probs[(unigram, b_p)]
      t_pos = t_posibilities[(unigram, b_p)]
      for t_p in t_pos:
        paux = t_probs[(unigram, b_p, t_p)]
        pt += paux * np.log2(paux)
      pb += p * pt
    h += pb * u_probs[unigram]
  
  return (-1 * h)


def perplexity(h):
  '''
  simple function that returns the perplexity given an entropy
  '''
  return np.power(2, h)


def smooth_countNgrams(l, x, inic, end=0):
    """
    From a list l (of words or pos), an inic position and an end position
    a tuple(U,B,T,Up,Bpx,Bpp,Tpxx,Tppx) of dics corresponding to unigrams, bigrams and trigrams are built
    """
    if end == 0:
        end = len(l)
    U = {}
    B = {}
    T = {}
    Up = {}
    Bpx = {}
    Bpp = {}
    Tpxx = {}
    Tppx = {}
    U[(l[inic][0])] = 1
    Up[(l[inic][1])] = 1
    if (l[inic+1][0]) not in U:
      U[(l[inic+1][0])] = 1
    else:
      U[(l[inic+1][0])] += 1
    if (l[inic+1][1]) not in Up:
      Up[(l[inic+1][1])] = 1
    else:
      Up[(l[inic+1][1])] += 1
    B[(l[inic][0],l[inic+1][0])] = 1
    Bpx[(l[inic][1],l[inic+1][0])] = 1
    Bpp[(l[inic][1],l[inic+1][1])] = 1
    for i in range(inic + 2,end):
      if (l[i][0]) not in U:
        U[(l[i][0])] = 1
      else:
        U[(l[i][0])] += 1
      if (l[i][1]) not in Up:
        Up[(l[i][1])] = 1
      else:
        Up[(l[i][1])] += 1
      if (l[i-1][0],l[i][0]) not in B:
        B[(l[i-1][0],l[i][0])] = 1
      else:
        B[(l[i-1][0],l[i][0])] += 1
      if (l[i-1][1],l[i][0]) not in Bpx:
        Bpx[(l[i-1][1],l[i][0])] = 1
      else:
        Bpx[(l[i-1][1],l[i][0])] += 1
      if (l[i-1][1],l[i][1]) not in Bpp:
        Bpp[(l[i-1][1],l[i][1])] = 1
      else:
        Bpp[(l[i-1][1],l[i][1])] += 1
      if (l[i-2][0],l[i-1][0],l[i][0]) not in T:
        T[(l[i-2][0],l[i-1][0],l[i][0])] = 1
      else:
        T[(l[i-2][0],l[i-1][0],l[i][0])] += 1
      if (l[i-2][1],l[i-1][0],l[i][0]) not in Tpxx:
        Tpxx[(l[i-2][1],l[i-1][0],l[i][0])] = 1
      else:
        Tpxx[(l[i-2][1],l[i-1][0],l[i][0])] += 1
      if (l[i-2][1],l[i-1][1],l[i][0]) not in Tppx:
        Tppx[(l[i-2][1],l[i-1][1],l[i][0])] = 1
      else:
        Tppx[(l[i-2][1],l[i-1][1],l[i][0])] += 1
    return (U,B,T,Up,Bpx,Bpp,Tpxx,Tppx)

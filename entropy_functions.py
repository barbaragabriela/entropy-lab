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


def bigram_model(unigrams, bigrams, u_probs, b_probs):
  h = 0.0
  for unigram in unigrams:
    pb = 0.0
    for bigram in bigrams:
      if bigram[0] == unigram:
        p = b_probs[bigram]
        pb += p * np.log2(p)
    h += pb * u_probs[unigram]

  return (-1 * h)

def trigram_model(unigrams, bigrams, trigrams, u_probs, b_probs, t_probs):
  h = 0.0
  for unigram in unigrams:
    pb = 0.0
    for bigram in bigrams:
      pt = 0.0
      if bigram[0] == unigram:
        p = b_probs[bigram]
        for trigram in trigrams:
          if (trigram[0],trigram[1]) == bigram:
            paux = t_probs[trigram]
            pt += paux * np.log2(paux)
        pb += p * pt
    h += pb * u_probs[unigram]
  
  return (-1 * h)

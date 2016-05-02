import numpy as np

def unigram_prob(unigram_dict, total_words):
  probs = {}
  for word in unigram_dict:
    probability = unigram_dict[word] / total_words
    probs[word] = probability

  return probs


def bigram_prob(bigram_dict, unigrams):
  probs = {}
  for bigram in bigram_dict:
    probability = bigram_dict[bigram] / float(unigrams[bigram[0]])
    probs[bigram] = probability

  return probs


def trigram_prob(trigram_dict, bigrams):
  probs = {}
  for trigram in trigram_dict:
    probability = trigram_dict[trigram] / float(bigrams[trigram[0],trigram[1]])
    probs[trigram] = probability

  return probs


def unigram_model(words, probs):
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

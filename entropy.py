import auxiliar
import entropy_functions as helper
# from nltk.model import NgramModel

def test():
  file = 'corpus/en2.txt'
  words = auxiliar.getWordsFromFile(file)
  total_words = float(len(words))
  
  U, B, T = auxiliar.countNgrams(words,0, 0)

  u_prob = helper.unigram_prob(U,total_words)
  b_prob = helper.bigram_prob(B,U)
  t_prob = helper.trigram_prob(T,B)




  u_h = helper.unigram_model(U, u_prob)
  print 'uh', u_h
  b_h = helper.bigram_model(U, B, u_prob, b_prob)
  print 'bh', b_h
  t_h = helper.trigram_model(U, B, T, u_prob, b_prob, t_prob)
  print 'th', t_h

test()

import auxiliar
import entropy_functions as helper

def compute_models():
  file = 'corpora/en.txt'
  words = auxiliar.getWordsFromFile(file)
  total_words = float(len(words))

  U, B, T = auxiliar.countNgrams(words,0, 0)

  u_prob = helper.unigram_prob(U,total_words)
  b_prob, b_posibilities = helper.bigram_prob(B,U)
  t_prob, t_posibilities = helper.trigram_prob(T,B)

  u_h = helper.unigram_model(U, u_prob)
  b_h = helper.bigram_model (U, b_posibilities, u_prob, b_prob)
  t_h = helper.trigram_model(U, b_posibilities, t_posibilities, u_prob, b_prob, t_prob)

  return u_h, b_h, t_h


def get_trigram_entropy(words, start, end):
  total_words = end
  U, B, T = auxiliar.countNgrams(words,start, end)

  u_prob = helper.unigram_prob(U,float(total_words))
  b_prob, b_posibilities = helper.bigram_prob(B,U)
  t_prob, t_posibilities = helper.trigram_prob(T,B)

  t_h = helper.trigram_model(U, b_posibilities, t_posibilities, u_prob, b_prob, t_prob)
  return t_h


def perplexity_trigram():
  file = 'corpora/taggedBrown.txt'
  words = auxiliar.getTaggedWordsFromFile(file)
  total_words = len(words)
  aux = total_words

  text = ['FULL CORPUS', 'HALF CORPUS', 'QUARTER CORPUS']
  i = 0
  while aux >= total_words / 4:
    h = get_trigram_entropy(words, 0, aux)
    perplexity = helper.perplexity(h)

    print text[i]
    print 'trigram entropy: H =', h
    print 'perplexity: ', perplexity

    i += 1
    aux = aux / 2

  print '\n'


def smooth_trigram():
  file = 'corpora/taggedBrown.txt'
  words = auxiliar.getTaggedWordsFromFile(file)
  total_words = len(words)
  aux = total_words

  text = ['FULL CORPUS', 'HALF CORPUS', 'QUARTER CORPUS']
  i = 0

  while aux >= total_words / 4:
    U, B, T, Up, Bpx, Bpp, Tpxx, Tppx = helper.smooth_countNgrams(words, 0, 0, aux)

    print text[i]
    # <X,Y,Z> Normal trigram
    u_prob = helper.unigram_prob(U,float(aux))
    b_prob, b_posibilities = helper.bigram_prob(B,U)
    t_prob, t_posibilities = helper.trigram_prob(T,B)

    t_h = helper.trigram_model(U, b_posibilities, t_posibilities, u_prob, b_prob, t_prob)
    pp = helper.perplexity(t_h)
    print '<X,Y,Z>'
    print 'trigram entropy: H =', t_h
    print 'perplexity: ', pp

    # <X',Y,Z> trigram
    u_prob = helper.unigram_prob(Up,float(aux))
    b_prob, b_posibilities = helper.bigram_prob(Bpx, Up)
    t_prob, t_posibilities = helper.trigram_prob(Tpxx,Bpx)

    t_h = helper.trigram_model(Up, b_posibilities, t_posibilities, u_prob, b_prob, t_prob)
    pp = helper.perplexity(t_h)
    print '<X\',Y,Z>'
    print 'trigram entropy: H =', t_h
    print 'perplexity: ', pp

    # <X',Y',Z> trigram
    u_prob = helper.unigram_prob(Up,float(aux))
    b_prob, b_posibilities = helper.bigram_prob(Bpp, Up)
    t_prob, t_posibilities = helper.trigram_prob(Tppx,Bpp)

    t_h = helper.trigram_model(Up, b_posibilities, t_posibilities, u_prob, b_prob, t_prob)
    pp = helper.perplexity(t_h)
    print '<X\',Y\',Z>'
    print 'trigram entropy: H =', t_h
    print 'perplexity: ', pp
    print '\n'

    i += 1
    aux = aux / 2


def run():
  option = ""
  while option != '0':
    print 'What do you want to see?'
    print '1) Compute Unigram, Bigram and Trigram model.'
    print '4) Brown corpus Perplexity'
    print '5) Smooth the trigram language model'
    print '0) Exit'
    print 'Choice: '
    option = raw_input()

    if option == '1':
      u_h, b_h, t_h = compute_models()
      print 'unigram entropy: H =', u_h
      print 'bigram entropy: H =', b_h
      print 'trigram entropy: H =', t_h
    elif option == '4':
      perplexity_trigram()
    elif option == '5':
      smooth_trigram()
    else:
      exit()


run()
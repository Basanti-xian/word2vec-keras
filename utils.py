import gensim, random
import numpy as np
from nltk.corpus import brown
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.sequence as seq
from gensim.models.keyedvectors import KeyedVectors


# output file
filename = 'vectors.txt'


def load_sentences_brown(nb_sentences=None):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: index2word (list of string)
    """

    print 'building vocab ...'

    if nb_sentences is None:
        sents = brown.sents()
    else:
        sents = brown.sents()[:nb_sentences]

    # I use gensim model only for building vocab
    model = gensim.models.Word2Vec()
    model.build_vocab(sents)
    vocab = model.wv.vocab

    # ids: list of (list of word-id)
    ids = [[vocab[w].index for w in sent
            if w in vocab and vocab[w].sample_int > model.random.rand() * 2**32]
           for sent in sents]

    return ids, model.wv.index2word



def get_indexing_word_vector(nb_sentences=None, max_num_words=None):

    # prepare text samples
    f = open('../data/imdb_labelled.txt', 'rU')

    print 'building vocab ...'

    if nb_sentences is None:
        texts = f.readlines()
    else:
        texts = f.readlines()[:nb_sentences]

    tokenizer = Tokenizer(num_words= max_num_words)
    tokenizer.fit_on_texts(''.join(sen) for sen in texts)
    # tokenizer.fit_on_texts(f.read()) # return str = '', out of memory?
    # https://docs.python.org/2/tutorial/inputoutput.html
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    return sequences, word_index



def skip_grams(sentences, window, vocab_size, nb_negative_samples=1):
    """
    calc `keras.preprocessing.sequence.skipgrams` for each sentence
    and concatenate those.

    :param sentences: list of (list of word-id)
    :return: concatenated skip-grams
    """

    print 'building skip-grams ...'

    def sg(sentence):
        return seq.skipgrams(sentence, vocab_size,
                             window_size=np.random.randint(window - 1) + 1,
                             negative_samples=nb_negative_samples)

    couples = []
    labels = []

    # concat all skipgrams
    for cpl, lbl in map(sg, sentences):
        couples.extend(cpl)
        labels.extend(lbl)

    return np.asarray(couples), np.asarray(labels)



def save_weights(model, index2word, vec_dim):
    """
    :param model: keras model
    :param index2word: list of string
    :param vec_dim: dim of embedding vector
    :return:
    """
    vec = model.get_weights()[1]
    f = open(filename, 'w')
    # first row in this file is vector information
    f.write(" ".join([str(len(index2word)), str(vec_dim)]))
    f.write("\n")
    for i, word in enumerate(index2word):
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, list(vec[i, :]))))
        f.write("\n")
    f.close()



def most_similar(positive=[], negative=[]):
    """
    :param positive: list of string
    :param negative: list of string
    :return:
    """
    vec = KeyedVectors.load_word2vec_format(filename, binary=False)
    for v in vec.most_similar_cosmul(positive=positive, negative=negative, topn=10):
        print(v)



def batch_generator(cpl, lbl,batch_size, nb_batch):

    # trim the tail
    garbage = len(lbl) % batch_size

    pvt = cpl[:, 0][:-garbage]
    ctx = cpl[:, 1][:-garbage]
    lbl = lbl[:-garbage]

    assert pvt.shape == ctx.shape == lbl.shape

    # epoch loop
    while 1:
        # shuffle data at beginning of every epoch (takes few minutes)
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(pvt)
        random.seed(seed)
        random.shuffle(ctx)
        random.seed(seed)
        random.shuffle(lbl)

        for i in range(nb_batch):
            begin, end = batch_size*i, batch_size*(i+1)
            # feed i th batch
            yield ([pvt[begin: end], ctx[begin: end]], lbl[begin: end])

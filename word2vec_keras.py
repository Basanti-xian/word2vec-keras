import sys
import utils
from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.merge import Dot
from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape, Flatten
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf8')



def make_word2vec_model(embedding_dim, num_words):
    '''
    embedding_dim: (int) embedding dimension
    num_words: (int) size of the vocabulary
    '''

    word_input = Input(shape=(1,), dtype='int32')
    context_input = Input(shape=(1,), dtype='int32')
    print '*'*10
    print 'input layer: ', word_input

    word_embedding = Embedding(num_words, embedding_dim)
    we = word_embedding(word_input)
    print 'word embedding layer: ', we

    context_embedding = Embedding(num_words, embedding_dim)
    ce = Reshape((embedding_dim, 1))(context_embedding(context_input))
    print 'context embedding layer: ', ce

    dots = Dot((1, 2))([ce, we])
    print 'merge layer: ', dots

    flat = Flatten()(dots)
    print 'flattern layer: ', flat
    print '*'*10

    acts = Activation('sigmoid')(flat)

    model = Model(inputs=[word_input, context_input], outputs=acts)
    model.compile('adam', loss='binary_crossentropy')
    return model


def main():

    sentences, index2word = utils.load_sentences_brown(800)

    # params
    nb_epoch = 3
    # learn `batch_size words` at a time
    batch_size = 128
    vec_dim = 64
    negsampling_num = 0.2
    # half of window
    window_size = 6
    vocab_size = len(index2word)

    print 'vocabulary length: ', vocab_size

    # create input
    couples, labels = utils.skip_grams(sentences, window_size, vocab_size, negsampling_num)
    print 'counter of positive samples and negative samples: ', Counter(labels)

    print 'shape of couples: ', couples.shape
    print 'shape of labels: ', labels.shape

    # metrics
    nb_batch = len(labels) // batch_size
    samples_per_epoch = batch_size * nb_batch

    # fit model
    model=make_word2vec_model(vec_dim, vocab_size)
    model.fit_generator(generator=utils.batch_generator(couples, labels, batch_size, nb_batch),
                        steps_per_epoch=samples_per_epoch,
                        epochs=nb_epoch)

    # save weights
    utils.save_weights(model, index2word, vec_dim)


    # eval using gensim
    print 'the....'
    utils.most_similar(positive=['the'])
    print 'all....'
    utils.most_similar(positive=['all'])
    print 'baby....'
    utils.most_similar(positive=['baby'])
    print 'first....'
    utils.most_similar(positive=['first'])


if __name__ == '__main__':
    main()
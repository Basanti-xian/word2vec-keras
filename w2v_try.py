from keras.layers import Input, Activation
from keras.layers.merge import Multiply, Dot
from keras.layers.core import Lambda, Reshape, Flatten
from keras.models import Model
from keras.layers.embeddings import Embedding
import utils


# load data
# - sentences: list of (list of word-id)
# - index2word: list of string
sentences, index2word = utils.load_sentences_brown(100)

# params
nb_epoch = 3
# learn `batch_size words` at a time
batch_size = 128
vec_dim = 64
negsampling_num = 0.2
# half of window
window_size = 8
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size, negsampling_num)
print 'shape of couples: ', couples.shape
print 'shape of labels: ', labels.shape

# metrics
nb_batch = len(labels) // batch_size
samples_per_epoch = batch_size * nb_batch

# graph definition (pvt: center of window, ctx: context)
input_pvt = Input(batch_shape=(batch_size, 1), dtype='int32')
input_ctx = Input(batch_shape=(batch_size, 1), dtype='int32')

embedded_pvt = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim,
                         input_length=1)(input_pvt)
embedded_pvt = Reshape((vec_dim, 1))(embedded_pvt)

embedded_ctx = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim,
                         input_length=1)(input_ctx)

'''
# merged = merge(inputs=[embedded_pvt, embedded_ctx],
#               mode=lambda x: (x[0] * x[1]).sum(-1),
#               output_shape=(batch_size, 1))
#embedded_pvt = tf.Session().run(embedded_pvt)
#embedded_ctx = tf.Session().run(embedded_ctx)
#merged = Lambda(lambda x: (x[0] * x[1]).sum(-1),
#                output_shape=(batch_size, 1))([embedded_pvt, embedded_ctx])

#merged = tf.reduce_sum(merged, -1)
'''

merged = Dot((1,2))([embedded_pvt, embedded_ctx])
print merged

flatten = Flatten()(merged)
print flatten
raw_input()

predictions = Activation('sigmoid')(flatten)


# build and train the model
model = Model(input=[input_pvt, input_ctx], output=predictions)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit_generator(generator=utils.batch_generator(couples, labels, batch_size, nb_batch),
                    steps_per_epoch=samples_per_epoch,
                    epochs=nb_epoch,
                    workers=1)

# save weights
utils.save_weights(model, index2word, vec_dim)

# eval using gensim
print 'the....'
utils.most_similar(positive=['the'])
print 'all....'
utils.most_similar(positive=['all'])
print 'she - he + him....'
utils.most_similar(positive=['she', 'him'], negative=['he'])

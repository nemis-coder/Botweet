""" Example to demostrate a GRU network to generate text.
UNAM IIMAS
Course:     Deep Learning
Professor:  Gibran Fuentes Pineda
Assistant:  Berenice Montalvo Lezama
"""

import datetime
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GRURNN:
  def __init__(self, alpha_size, rnn_layers_size, rnn_layers_num, sess):
    """ Creates the model """
    self.alpha_size = alpha_size
    self.rnn_layers_size = rnn_layers_size
    self.rnn_layers_num = rnn_layers_num
    self.sess = sess
    self.rnn_state_size = self.rnn_layers_size * self.rnn_layers_num
    self.def_input()
    self.def_params()
    self.def_model()
    self.def_output()
    self.def_loss()
    self.def_summaries()

  def def_input(self,):
    """ Defines inputs """
    with tf.name_scope('input'):
      self.X = tf.placeholder(tf.uint8,
        [None, None], name='X')
      self.YT = tf.placeholder(tf.uint8,
        [None, None], name='YT')
      self.init_state = tf.placeholder(tf.float32,
        [None, self.rnn_state_size], name='init_state')
      # [batch_size, seq_size] -> [batch_size, seq_size, alpha_size]
      self.X1h = tf.one_hot(self.X,
        self.alpha_size, 1.0, 0.0, name='X1h')
      self.YT1h = tf.one_hot(self.YT,
        self.alpha_size, 1.0, 0.0, name='YT1h')

  def def_params(self):
    """ Defines model parameters """
    with tf.name_scope('params'):
      # GRU
      self.rnn_layers = [rnn.GRUCell(self.rnn_layers_size)
        for _ in range(self.rnn_layers_num)]
      # FC
      initializer = tf.contrib.layers.xavier_initializer()(
          [self.rnn_layers_size, self.alpha_size])
      self.W = tf.Variable(initializer, name='W')
      self.B = tf.Variable(tf.random_normal(
        [self.alpha_size], stddev=0.01), name='B')

  def def_model(self):
    """ Defines the model """
    with tf.name_scope('model'):
      # chains RNN layers
      self.rnn = rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=False)
      # unrolls RNN layers
      H, self.next_state = tf.nn.dynamic_rnn(self.rnn, self.X1h,
        initial_state=self.init_state, dtype=tf.float32)
      # [batch_size, seq_size, rnn_layers_size] -> [batch_size*seq_size, rnn_layers_size]
      HR = tf.reshape(H, [-1, self.rnn_layers_size], name ='HR')
      # FC Flat
      self.YP1hF = tf.matmul(HR, self.W) + self.B
      self.YS1hF = tf.nn.softmax(self.YP1hF)
      # [batch_size*seq_size, rnn_layers_size] -> [batch_size, seq_size, rnn_layers_size]
      seq_size = tf.shape(H)[1]
      Y_shape = [-1, seq_size, self.alpha_size]
      self.YP1h = tf.reshape(self.YS1hF, Y_shape, name='YP1h')

  def def_output(self):
    with tf.name_scope('output'):
      self.YP = tf.argmax(self.YP1h, name='YP')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):
      # cross entropy
      YT1hF = tf.reshape(self.YT1h, [-1, self.alpha_size])
      self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=self.YP1hF, labels=YT1hF), name='cross_entropy')

  def def_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries'):
      tf.summary.scalar('loss', self.loss)
      #~ tf.summary.histogram('W', self.W)
      #~ tf.summary.histogram('B', self.B)
      self.summary = tf.summary.merge_all()

  def train(self, data, batch_size, seq_size, train_iters,
      eval_step=None, learning_rate=0.003):
    """ Trains the model """
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    # initialize variables (params)
    self.sess.run(tf.global_variables_initializer())
    # writers for TensorBorad
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    model_name  = 'graphs/11_gru_{}'.format(timestamp)
    train_writer = tf.summary.FileWriter(model_name)
    train_writer.add_graph(self.sess.graph)
    # rnn state
    init_state = np.zeros([batch_size, self.rnn_state_size])

    # training loop
    for i in range(train_iters):
      X, Y = data.next_train(batch_size, seq_size)

      if eval_step and not i % eval_step:

        feed_dict = {self.X: X, self.YT: Y, self.init_state: init_state}
        fetches = [self.loss, self.summary]
        loss, summary = self.sess.run(fetches, feed_dict)

        train_writer.add_summary(summary, i)
        msg = "{:5d}. loss: {:6.4f}"
        msg = msg.format(i, loss)
        print(msg)

      feed_dict = {self.X: X, self.YT: Y, self.init_state: init_state}
      fetches = [optimizer, self.next_state]
      _, next_state = self.sess.run(fetches, feed_dict)

      init_state = next_state

    self.last_state = init_state

    X, Y = data.next_train(batch_size, seq_size)

    feed_dict = {self.X: X, self.YT: Y, self.init_state: init_state}
    fetches = [self.loss, self.summary]
    loss, summary = self.sess.run(fetches, feed_dict)

    train_writer.add_summary(summary, i)
    msg = "{:5d}. loss: {:6.4f}"
    msg = msg.format(i, loss)
    print(msg)

  def run_step(self, x, reset_state=False):
    ## Reset the initial state of the rnn
    if reset_state:
      init_state = np.zeros([self.rnn_state_size])
    else:
      init_state = self.last_state

    feed_dict = {self.X:[[x]], self.init_state:[init_state]}
    fetches = [self.YP, self.YP1h, self.next_state]
    pred, pred1h, next_state = self.sess.run(fetches, feed_dict)

    self.last_state = next_state[0]

    return pred[0][0], pred1h[0][0]


class DatatReader():

  def __init__(self, path):
    with open(path, 'r') as f:
      self.text = f.read().lower()
    self.alpha = sorted(list(set(self.text)))
    self.alpha_dict = {c:i for i, c in enumerate(self.alpha)}

  def next_train(self, batch_size, seq_size):
    space = range(len(self.text) - seq_size - 1)
    sampled = random.sample(space, batch_size)
    X, Y = [], []
    for s in sampled:
      seq = self.text[s:s+seq_size]
      X.append(self.encode(seq))
      seq = self.text[s+1:s+1+seq_size]
      Y.append(self.encode(seq))
    return X, Y

  def decode(self, text):
    return [self.alpha[c] for c in text]

  def encode(self, text):
    return [self.alpha_dict[c] for c in text]


def run():

  path = 'Corpus_Tweets_vicentefoxque.txt'#'shakespeare.txt'
  data = DatatReader(path)

  alpha_size  = len(data.alpha)
  rnn_layers_size   = alpha_size * 2
  rnn_layers_num  = 3
  batch_size  = 64
  seq_size    = 32
  train_iters = 500

  text_seed = "muro de trump"
  composition_size = 500

  with tf.Session() as sess:

    model = GRURNN(alpha_size, rnn_layers_size, rnn_layers_num, sess)

    print("Training:")
    model.train(data, batch_size, seq_size, train_iters, eval_step=10)

    composition = []
    for i, c in enumerate(text_seed):
      enc = data.encode(c)[0]
      composition.append(enc)
      _, dist = model.run_step(enc, i==0)

    for i in range(composition_size):
      c = np.random.choice(data.alpha, p=dist)
      enc = data.encode(c)[0]
      composition.append(enc)
      _, dist = model.run_step(enc)

    composition = data.decode(composition)
    composition = ''.join(composition)
    print("\nOur Composition:")
    print('<<<%s>>>' % composition)

  save_path = saver.save(sess, "model.ckpt")
  print("Model saved in file: %s" % save_path)
    
  return 0


def main(args):
  run()
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))

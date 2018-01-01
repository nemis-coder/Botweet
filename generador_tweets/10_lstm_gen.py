""" Example to demostrate a simple LSTM network to generate text.
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

import urllib.request
import json

import tweepy as tp
import re
import pickle as pk

import unicodedata

consumer_key = "TM6dANbzBL5LzpXox05KJePwy"
consumer_secret = "HV3nuTA82M1tjTE2sKTeKPZ8kWDi5r4EgUGE4ogroinPcmfPqe"
access_key = "921130974569811968-RcDUIosERaFWnHHYw5C6OSq4aAGYdKi"
access_secret = "AOPClE7TtSHMuO8Jbs4TNhdxfEfPK5s3cbwnJpCE3s2v3"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SLSTM:
  def __init__(self, alpha_size, lstm_size, lstm_depth, sess):
    """ Creates the model """
    self.alpha_size = alpha_size
    self.lstm_size = lstm_size
    self.lstm_depth = lstm_depth
    self.sess = sess
    self.lstm_state_size = self.lstm_size * self.lstm_depth * 2
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
        [None, self.lstm_state_size], name='init_state')
      # [batch_size, seq_size] -> [batch_size, seq_size, alpha_size]
      self.X1h = tf.one_hot(self.X,
        self.alpha_size, 1.0, 0.0, name='X1h')
      self.YT1h = tf.one_hot(self.YT,
        self.alpha_size, 1.0, 0.0, name='YT1h')

  def def_params(self):
    """ Defines model parameters """
    with tf.name_scope('params'):
      # LSTM
      self.lstm_layers = [rnn.BasicLSTMCell(self.lstm_size,
        state_is_tuple=False) for _ in range(self.lstm_depth)]
      # FC
      self.W = tf.Variable(tf.random_normal(
        [self.lstm_size, self.alpha_size], stddev=0.01), name='W')
      self.B = tf.Variable(tf.random_normal(
        [self.alpha_size], stddev=0.01), name='B')

  def def_model(self):
    """ Defines the model """
    with tf.name_scope('model'):
      # chains LSTM layers
      self.lstm = rnn.MultiRNNCell(self.lstm_layers, state_is_tuple=False)
      # unrolls LSTM layers
      H, self.next_state = tf.nn.dynamic_rnn(self.lstm, self.X1h,
        initial_state=self.init_state, dtype=tf.float32)
      # [batch_size, seq_size, lstm_size] -> [batch_size*seq_size, lstm_size]
      HR = tf.reshape(H, [-1, self.lstm_size], name ='HR')
      # FC Flat
      self.YP1hF = tf.matmul(HR, self.W) + self.B
      self.YS1hF = tf.nn.softmax(self.YP1hF)
      # [batch_size*seq_size, lstm_size] -> [batch_size, seq_size, lstm_size]
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
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
    # initialize variables (params)
    self.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # writers for TensorBorad
    timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    model_name  = 'graphs_2/10_lstm_{}'.format(timestamp)
    train_writer = tf.summary.FileWriter(model_name)
    train_writer.add_graph(self.sess.graph)
    # lstm state
    init_state = np.zeros([batch_size, self.lstm_state_size])

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

    #saver.save(self.sess,"my_model")
    train_writer.add_summary(summary, i)
    msg = "{:5d}. loss: {:6.4f}"
    msg = msg.format(i, loss)
    saver.save(self.sess,"model/my_model")
    print(msg)

  def run_step(self, x, reset_state=False):
    ## Reset the initial state of the LSTM
    if reset_state:
      init_state = np.zeros([self.lstm_state_size])
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

def get_gif(semilla):
  flag = False
  f = open('my.gif','wb') 
  ur = "https://api.giphy.com/v1/gifs/search?api_key=UvWIQAUpscOXqQe8Vioxm701TCROf4Up&q="+semilla+"&limit=5&offset=0&rating=G&lang=es"
  with urllib.request.urlopen(ur) as url:
    s = url.read().decode('utf8')
    data = json.loads(s)
    #print ((data['data'][0]['images']['fixed_height']['url']))
    ur= ((data['data'][0]['images']['fixed_height']['url']))
    if (len(data['data'])>0):
      flag= True
    urllib.request.urlretrieve (ur, 'my.gif')
  f.close()
  return flag
  
def run(text_seed):

  path = '../corpus_tweets/Corpus_Tweets_vicentefoxque.txt' #'shakespeare.txt'
  data = DatatReader(path)

  alpha_size  = len(data.alpha)
  #print ("The alpha_size is",alpha_size)
  lstm_size   = 512#alpha_size * 2
  lstm_depth  = 3
  batch_size  = 50
  seq_size    = 32
  train_iters = 6000

  #text_seed = "mexico " #"grandioso dia de compartir ideas y conocer nuevas visiones del mundo"
  #text_seed = "venezuela  "
  #composition_size = 140
  composition_size = 140
  
  
  with tf.Session() as sess:

  
    
    model = SLSTM(alpha_size, lstm_size, lstm_depth, sess)

    #print("Training:")
    #model.train(data, batch_size, seq_size, train_iters, eval_step=10)  


    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint("../model/"))
    #sess.run(tf.global_variables_initializer())

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
  return composition

def get_tweets_seed(nombre):
    url_ = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
    has_ = r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)'
    refe_ = r'(?:@[\w_]+)'
    dict_stop = pk.load(open("dict_stop.p","rb"))
    
    auth = tp.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tp.API(auth)

    tweets = api.user_timeline(nombre,count=5)
    texto_sucio = []
    for tweet in tweets:
        texto = tweet.text
        if 'RT @' not in texto:
            texto_limp = re.sub(url_, '', texto)
            texto_limp = re.sub(has_, '', texto_limp)
            texto_limp = re.sub(refe_, '', texto_limp)
            texto_limp.replace('\n', ' ')
            texto_to_list = texto_limp.split()
            texto_sucio.extend(texto_to_list)
    texto_limpio = [] 
    for palabra in texto_sucio:
        if palabra not in dict_stop and (((palabra[0]).lower())+palabra[1:]) not in dict_stop:
            texto_limpio.append(palabra)
    return (texto_limpio)
    
"""            
def bot():
    auth = tp.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tp.API(auth)
    #api.update_status("Test")
    return (semilla)
"""

def main(args):
  try:
    semillas = get_tweets_seed(nombre="Pajaropolitico")
    semilla = random.choice(semillas)
    semilla = ((unicodedata.normalize('NFKD', semilla).encode('ascii','ignore')).decode("ascii"))
    print ("semilla",semilla)
    fl = get_gif(semilla.lower())
    composition = run(semilla.lower())
    auth = tp.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tp.API(auth)
    r = random.randint(0,1)
    print (r)
    if r==1 and fl:
      api.update_with_media(filename='my.gif',status=composition[0:140])
    else:
      api.update_status(composition[0:140])
  except:
    print ("Here")
    semillas = get_tweets_seed(nombre="Pajaropolitico")
    semilla = random.choice(semillas)
    composition = run(semilla.lower())
    auth = tp.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tp.API(auth)
    api.update_status(composition[0:140])
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))

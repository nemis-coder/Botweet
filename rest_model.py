import tensorflow as tf

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


path = 'Corpus_Tweets_felipecalderon.txt' #'shakespeare.txt'
data = DatatReader(path)
  
text_seed= "mexico "
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model/my_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
    # print (sess.run('input/X:0'))
    #my_variable = tf.get_variable("X", [1, 2, 3])
    composition = []
    graph = tf.get_default_graph()
    X = tf.get_variable("input/X:0",initializer=tf.constant([[39]]))

    
    

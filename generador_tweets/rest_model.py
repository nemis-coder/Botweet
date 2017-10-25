import tensorflow as tf
import numpy as np

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
    graph = tf.get_default_graph()
    # print (sess.run('input/X:0'))
    X = graph.get_tensor_by_name("input/X:0")
    init_state = graph.get_tensor_by_name("input/init_state:0")

    #composition = []
    #graph = tf.get_default_graph()

    #init_state = np.zeros([512*3*2])

    feed_dict = {X:[[39]],init_state:[np.zeros([(512*3*2)])]}
    op_to_restore = graph.get_tensor_by_name("output/YP:0")
    next_state = graph.get_tensor_by_name("output/next_state:0")
    YP, state = sess.run(op_to_restore,feed_dict)

    feed_dict = {X:[[39]],init_state:state}
    YP, state = sess.run(op_to_restore,feed_dict)
    print(YP, state)

    #print (sess.run(op_to_restore,feed_dict))
    #pred, pred1h, next_state = sess.run(fetches, feed_dict)
    #X = tf.get_variable("input/X:0",initializer=tf.constant([[39]]))
    #op = sess.graph.get_operations()
    #for m in op:
     # print (m.values)

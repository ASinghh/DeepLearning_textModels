from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
url = 'http://mattmahoney.net/dc/'

url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data
  
text = read_data(filename)
print('Data size %d' % len(text))


valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

Vocab_size = 27 ##26 letter + 1 for free space
first_letter = ord(string.ascii_lowercase[0])

text_size_train = len(train_text)
batch_size = 64
num_unrollings = 10
segment = text_size_train // batch_size
cursor_tr = [ offset * segment for offset in range(batch_size)]




                 
def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0
                 
def char2code(list):
    p = 0
    for i in range(len(list)):
        p = p + (27**(len(list)-i-1)*char2id(list[i]))
        
    return p

def hotenco(j,n_gram): 
    k = []
    for i in j:
        p = []
        for b in range(Vocab_size**n_gram):
            if b == i :
                p.append(1)
            else:
                p.append(0)
        k.append(p)
    return np.asarray(k)
def shift(j,k):
    a = hot2char(j)
    b = hot1char(k)
    c = a[1] + b[0]
    p = char2code(c)
    v = hotenco([p],2)
    return v
        
def batch_generator():
    train_x = []
    train_y =[]
    for i in range(batch_size):
        train_x.append(char2code(train_text[cursor_tr[i]] + train_text[cursor_tr[i]+1]))
        train_y.append(char2code(train_text[cursor_tr[i]+2]))
        cursor_tr[i] = (cursor_tr[i] + 1) % text_size_train
    return train_x , train_y
def hot2char(vec): 
    a = [0,0]
    a[0] = id2char(np.argmax(vec)//27)
    a[1] = id2char(np.argmax(vec)%27)
    return a
def hot1char(vec): 
    a = [0]
    a[0] = id2char(np.argmax(vec)%27)
    
    return a


def batches(size):
    train_batches = []
    label_batches = []
    for i in range(size):
        d,l = batch_generator()
        train_batches.append(hotenco(d,2))
        label_batches.append(hotenco(l,1))
    return train_batches , label_batches

###########################################################################
    

    
num_nodes = 512
embedding_size = 128
graph = tf.Graph()
with graph.as_default():
    
      # Parameters:
  # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
    cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes,Vocab_size ], -0.1, 0.1))
    b = tf.Variable(tf.zeros([Vocab_size]))
    
    embeddings = tf.Variable(tf.random_uniform([Vocab_size**2, embedding_size], -1.0, 1.0))
    
    
    def lstm_cell(i, o, state):

        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state
    
    
       
    train_inputs = list()
    for _ in range(10):
        train_inputs.append(
        tf.placeholder(tf.float32, shape=[batch_size,Vocab_size**2]))
    
    train_labels = list()
    for _ in range(10):
        train_labels.append(
        tf.placeholder(tf.float32, shape=[batch_size,Vocab_size]))
        
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(tf.matmul(i,embeddings), output, state)
        outputs.append(output)
    with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                 labels=tf.concat(train_labels, 0), logits=logits))
        
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
              5.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)
        
    train_prediction = tf.nn.softmax(logits)
    sample_input = tf.placeholder(tf.float32, shape=[1,Vocab_size**2 ])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
    tf.matmul(sample_input,embeddings), saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
    

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    t,l = batches(10)
    feed_dict = dict()
    for i in range(10):
      feed_dict[train_inputs[i]] = t[i]
      feed_dict[train_labels[i]] = l[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
     
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      if step % (summary_frequency * 2) == 0:
       
        print('=' * 80)
        pred_test =[] 
        test = []  
        for _ in range(5):
            k = [0]*729
            k[np.random.randint(0, 728)]   = 1  
            feed = np.reshape(np.asarray(k),(1,729))
            pred_test.append(feed)
            sentence = hot2char(k)[0] + hot2char(k)[1]
            reset_sample_state.run()
            
            for _ in range(100):
                prediction = sample_prediction.eval({sample_input: feed})
                
                feed = shift(feed, prediction)
                pred_test.append(feed)
                
                sentence += hot1char(prediction)[0]
                test.append(hot1char(prediction)[0])
            print(sentence)
        print('=' * 80)
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

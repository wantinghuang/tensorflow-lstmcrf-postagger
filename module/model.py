import tensorflow as tf
import json

def padding(data,per_size,max_step):
    # data is a list with length of sentance size, originally 2d
    zeros = [0.0 for i in range(per_size)]
    try:
        pad = [x.tolist() for x in data] # turn word embedding array to list
    except:
        pad = data.copy()
    if len(data) < max_step:
        while (len(pad) < max_step):
            pad.append(zeros)
    elif len(data) > max_step:
        pad = pad[:max_step]
        #print('input exceeds max_step!')
    else:
        pass
    #print(len(pad))
    return pad

def minibatch(data, batch_size):
    batch_ls=[]
    for i in range(int(len(data)/batch_size)):
        batch_ls.append(data[batch_size*i:batch_size*(i+1)])
    if len(data) % batch_size != 0 :
        res_size = len(data) % batch_size
        batch_ls.append(data[-1*res_size:])
    return batch_ls

def multiminibatch(data, batch_size): 
    if len(data) >1:
        batch_ls=[[]]*len(data)
        for j in range(len(data)):
            batch_ls[j] = minibatch(data[j],batch_size)
    elif len(data)==1:
        return minibatch(data, batch_size)
    return batch_ls

def readdata(filename, embedding_type='fasttext'):
    # filename = 'training_dt'+str(ix)+'.json' for ix in range(10)
    # filename = 'testing_dt.json'
    if embedding_type!='word2vec' and embedding_type!='fasttext':
        print("No such embedding type")
        return
    X_fasttext = []
    X_word2vec = []
    y_onehotlabel=[]
    X_seg=[]
    y_label=[]
    #   raw_sentence=[]
    with open(filename,'r') as f:
        dt = json.load(f) 
    for i in dt.keys():
        X_fasttext.append(dt[i]['fasttext'])
        X_word2vec.append(dt[i]['word2vec'])
        y_onehotlabel.append(dt[i]['onehot_label'])
        X_seg.append(dt[i]['original_sentence'])
        y_label.append(dt[i]['original_label'])
        #raw_sentence.append(dt[i]['origin'].strip(' '))
    if embedding_type=='word2vec':
        #return X_word2vec, y_onehotlabel, raw_sentence
        return X_word2vec, y_onehotlabel, X_seg, y_label
    elif embedding_type=='fasttext':
        #return X_fasttext, y_onehotlabel, raw_sentence
        return X_fasttext, y_onehotlabel, X_seg, y_label


class Tagger(object):
    def __init__(self, config):
        self.max_step = config.max_step
        self.n_classes = config.n_classes
        self.n_inputs = config.n_inputs
        self.n_hidden_units = config.n_hidden_units

        # tf.placeholder
        self.x = tf.placeholder(tf.float32, [None, self.max_step, self.n_inputs], name='input') # (batch_size, max_time_step, in)
        self.y = tf.placeholder(tf.float32, [None, self.max_step, self.n_classes], name='label') # (batch_size, max_time_step, out)
        self.length = tf.placeholder(tf.int32, [None], name='sentence_length')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate') 
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')

        # tf.Variable
        self.weights = {
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        self.biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes]))
        }
        #self.wordembedding = tf.Variable(tf.random_normal([self.n_inputs, self.n_inputs]))
        #inputs = tf.reshape(self.x,[-1,self.n_inputs])
        #inputs = tf.matmul(inputs,self.wordembedding)
        #inputs = tf.reshape(inputs,[-1,self.max_step,self.n_inputs])
        #cell = tf.contrib.rnn.LSTMCell(self.n_hidden_units)
        #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.dropout)
        #outputs, state = tf.nn.dynamic_rnn(cell, inputs,
        #                                   sequence_length=self.length,
        #                                   dtype=tf.float32)
        #outputs = tf.reshape(outputs,[-1,self.n_hidden_units])
        #results = tf.matmul(outputs, self.weights['out']) + self.biases['out']
        #self.logits = tf.reshape(results,[-1,self.max_step,self.n_classes])
        self.logits = self.RNN(self.x, self.weights, self.biases, self.length, self.dropout)

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        mask = tf.sequence_mask(self.length)
        losses = tf.boolean_mask(losses, mask)
        self.__loss = tf.reduce_mean(losses)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.__train_op = optimizer.minimize(self.__loss)
        
        # Evaluate model (with test logits, for dropout to be disabled)
        mask2 = tf.sequence_mask(self.length, maxlen = self.max_step)
        mask_logit = tf.boolean_mask(self.logits, mask2)
        self.__prediction = tf.nn.softmax(mask_logit) # prediction = mask_logit
        mask_label = tf.boolean_mask(self.y, mask2)
        # tf.argmax: be careful to choose the right axis to find argmax
        correct_pred = tf.equal(tf.argmax(self.__prediction, axis=1), tf.argmax(mask_label, axis=1)) # correct_pred: all labels with true length in minibatch combine together
        self.__correct_index = tf.cast(correct_pred, tf.float32)
        self.__accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
    @property
    def loss(self):
        return self.__loss
    
    @property
    def train_op(self):
        return self.__train_op
    
    @property
    def prediction(self):
        return self.__prediction
        
    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def correct_index(self):
        return self.__correct_index
    
    
    def RNN(self, X, weights, biases,seq_length,dropout):
        inputs = tf.reshape(X,[-1,self.n_inputs])
        inputs = tf.matmul(inputs,self.wordembedding)
        inputs = tf.reshape(inputs,[-1,self.max_step,self.n_inputs])
        # create a BasicRNNCell
        #basicrnn = tf.contrib.rnn.BasicRNNCell(n_hidden_units)
        cell = tf.contrib.rnn.LSTMCell(self.n_hidden_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
        #initial_state = basicrnn.zero_state(batch_size, dtype=tf.float32)
        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        # 'state' is a tensor of shape [batch_size, cell_state_size]
        outputs, state = tf.nn.dynamic_rnn(cell, inputs,
        #                                   initial_state=initial_state,
                                           sequence_length=seq_length,
                                           dtype=tf.float32)
    
        outputs = tf.reshape(outputs,[-1,self.n_hidden_units])
        results = tf.matmul(outputs, weights['out']) + biases['out']
        results = tf.reshape(results,[-1,self.max_step,self.n_classes])   
        return results           
      
    def pre_process(self, batch_xs,batch_ys):
        inputs=[]
        seq_length=[]
        labels=[]
        for data,label in zip(batch_xs,batch_ys): # data is a sentence without padding
            inputs.append(padding(data,self.n_inputs,self.max_step))
            labels.append(padding(label,self.n_classes,self.max_step))
            seq_length.append(min(len(data),self.max_step))
            
        return inputs, labels, seq_length
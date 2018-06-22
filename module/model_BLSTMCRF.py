import tensorflow as tf
import json

class BCRFTagger(object):
    def __init__(self, config):
        self.max_step = config.max_step
        self.n_classes = config.n_classes
        #embedding_size = config.embedding_size
        self.n_inputs = config.n_inputs
        self.n_hidden_units = config.n_hidden_units
        #batch_size = config.batch_size
        #num_epoch = config.num_epoch
        #num_layers = 2
        
        # tf.placeholder
        self.x = tf.placeholder(tf.float32, [None, self.max_step, self.n_inputs], name='input') # (batch_size, max_time_step, in)
        self.y = tf.placeholder(tf.float32, [None, self.max_step, self.n_classes], name='label') # (batch_size, max_time_step, out)
        self.length = tf.placeholder(tf.int32, [None], name='sentence_length')
        self.dropout = tf.placeholder(tf.float32, name='dropout_rate') 
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')

        # tf.Variable for biRNN
        self.bi_weights = {
            'in': tf.Variable(tf.random_normal([self.n_inputs, 2*self.n_hidden_units])),
            'out': tf.Variable(tf.random_normal([2*self.n_hidden_units, self.n_classes]))
        }
        self.bi_biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[2*self.n_hidden_units, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes]))
        }
        
        self.logits = self.biRNN(self.x, self.bi_weights, self.bi_biases,self.length, self.dropout)     
        y_t = tf.argmax(self.y, 2)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, y_t, self.length)


        # Define loss and optimizer
        #losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        #mask = tf.sequence_mask(self.length)
        #losses = tf.boolean_mask(losses, mask)
        self.__loss = tf.reduce_mean(-log_likelihood)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.__train_op = optimizer.minimize(self.__loss)
        
        # Evaluate model (with test logits, for dropout to be disabled)
        mask2 = tf.sequence_mask(self.length, maxlen = self.max_step)

        #softmaxlogits = tf.nn.softmax(self.logits)
        #self.__tags_seq, tags_score = tf.contrib.crf.crf_decode(softmaxlogits, transition_params, self.length)         
        self.__tags_seq, tags_score = tf.contrib.crf.crf_decode(self.logits, transition_params, self.length) 
        
        y_t = tf.cast(y_t,tf.int32)
        
        self.__prediction = tf.boolean_mask(self.__tags_seq, mask2) # self.__prediction = tf.nn.softmax(mask_logit) # prediction = mask_logit
        mask_label = tf.boolean_mask(y_t, mask2)#mask_label = tf.boolean_mask(self.y, mask2)
        # tf.argmax: be careful to choose the right axis to find argmax
        correct_pred = tf.equal(self.__prediction, mask_label) #correct_pred = tf.equal(tf.argmax(self.__prediction, axis=1), tf.argmax(mask_label, axis=1)) # correct_pred: all labels with true length in minibatch combine together
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
    @property
    def tags_seq(self):
        return self.__tags_seq
    
    
    def biRNN(self, X, weights, biases,seq_length, dropout):
        inputs = X
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=1.0 - dropout, output_keep_prob=1.0 - dropout)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=1.0 - dropout, output_keep_prob=1.0 - dropout)
    
        state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=seq_length, dtype=tf.float32)
        state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
        state_shape = state_outputs.get_shape()
        outputs = tf.reshape(state_outputs,[-1,2*self.n_hidden_units])
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
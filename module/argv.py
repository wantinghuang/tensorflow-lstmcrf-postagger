# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:03:17 2018

@author: wanting_huang
"""
import tensorflow as tf
import os
import numpy as np
import json
from gensim.models import word2vec
import parameter
import model
import model_biRNN
import model_LSTMCRF
import model_BLSTMCRF
import argparse
import string
    
def POStagger(sess, config, tagger, flag, data_dir):
    with open(data_dir+'posdt.json','r') as f:
        posdt = json.load(f)
    inv_map = dict(zip(posdt.values(), posdt.keys()))
    
    log = open(os.path.join(data_dir,file_to_save_model,'output.txt'),'a',encoding='utf-8')
    '''
    Function: 自己輸入input句子，回傳POS tag
    所以function要可以接受輸入的句子，斷詞、轉成word vector、
    放入已經訓練好的模型產生預測結果、回傳對應的tag
    input: 標準輸入數據，回傳str
    '''
    ## call out parameters
    n_hidden_units = config.n_hidden_units       # neurons in hidden layer
    n_classes = config.n_classes
    n_inputs = config.n_inputs
    max_step = config.max_step # max_time_step
    
    ## word segementation
    

    '''
    # adding lexicon to jieba traditional chinese dictionary
    # chinese
    jieba.set_dictionary('../zh/dict.txt.big')
    jieba.load_userdict("../zh/segmentdict.txt") # add the corpus to jieba lexicon 
    '''

    ## get input sentence and show results
    # let user keep using it in while loop
    # when leaving, type "q".
    while True:
        try:
            inputsentence = input('write down sentences and see the POS tags: (exit: enter "q")\n')
        except:
            continue
        if inputsentence == 'q':
            break
        '''
        # chinese
        import jieba
        seg_list = jieba.cut(inputsentence, cut_all=False)
        '''
        # english
        for c in string.punctuation:
        	inputsentence=inputsentence.replace(c,' '+c)
        seg_list = inputsentence.split(' ')

        sentence = [word for word in seg_list]
        model_ted = word2vec.Word2Vec.load(data_dir+"fasttext.model")
        #sentence_vector = [model_ted.wv[word].tolist() if model_ted.wv[word].tolist() else [0]*100 for word in sentence]
        # model = word2vec.Word2Vec.load("word2vec.model")
        sentence_vector = []
        for word in sentence:
            try:
                sentence_vector.append(model_ted.wv[word].tolist())
            except:
                sentence_vector.append([0]*n_inputs)
        inputs=[]
        seq_length=[]
        for data in [sentence_vector] : # data is a sentence without padding!!! 
            # watch out for what happened if there is only one sentence. Then it should be a list with one element.
            inputs.append(model.padding(data,n_inputs,max_step))
            seq_length.append(min(len(data),max_step)) 
    
        pred = sess.run(tagger.prediction,feed_dict={tagger.x: inputs, tagger.length: seq_length, tagger.dropout: 0.0})
        if flag == '1' or flag == '2':
            tag_ls = np.argmax(pred,axis=1) #if use model LSTM/BLSTM, pred return a onehot encoding vector
        elif flag == '3' or flag == '4':
            tag_ls = pred #if use model combine CRF, pred return a label

        result=''
        for i in range(len(tag_ls)):
            result = result + sentence[i] + '_' + inv_map[tag_ls[i]] + ' '
        print(result)
        log.write(result+'\n');log.flush()
        #return result
    log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", type=str, default='../en/', dest="data_dir", help="specify your data directory")
    parser.add_argument("--model_path", type=str, default='lstmcrf0621', dest="file_to_save_model",help="name the file to save your model")
    parser.add_argument("--model_type", type=str, default='3', dest="flag", help="the model to be trained \n1: LSTM-RNN \n2: BiLSTM-RNN \n3: LSTM+CRF \n4: BLSTM+CRF \n")
    arg=parser.parse_args()
    data_dir = arg.data_dir
    file_to_save_model = arg.file_to_save_model
    flag = arg.flag

    config = parameter.Config()

    if flag =='1':
        tagger = model.Tagger(config=config)
    elif flag =='2': 
        tagger = model_biRNN.biRNNTagger(config=config)
    elif flag =='3':
        tagger = model_LSTMCRF.CRFTagger(config=config)
    elif flag =='4':
    	tagger = model_BLSTMCRF.CRFTagger(config=config)
    else:
        print("No such model")
        exit()

    init = tf.global_variables_initializer()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(data_dir,file_to_save_model)))
        POStagger(sess, config, tagger,flag, data_dir)
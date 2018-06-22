# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:02:54 2018

@author: wanting_huang
"""
import tensorflow as tf
import os
from sklearn.utils import shuffle
import timeit
import json
import model
import parameter
import model_biRNN
import model_LSTMCRF
import model_BLSTMCRF
import argparse
         
def train(flag, file_to_save_model, data_dir):
    num_of_training_dt = 10
    config = parameter.Config()
    if flag == '1':
        tagger = model.Tagger(config=config)
    elif flag == '2':
        tagger = model_biRNN.biRNNTagger(config=config)
    elif flag == '3':
        tagger = model_LSTMCRF.CRFTagger(config=config)
    elif flag == '4':
        tagger = model_BLSTMCRF.BCRFTagger(config=config)
    else:
    	print("No such model!")
    	exit()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)  
    sess.run(init)
    
    # create the path to save the model
    if not os.path.exists( os.path.join('.',file_to_save_model) ):
        os.mkdir(os.path.join('.',file_to_save_model))
        
    saver = tf.train.Saver(max_to_keep=3)
    trainconfig = parameter.TrainConfig()
    batch_size = trainconfig.batch_size
    num_epoch = trainconfig.num_epoch
    eps = trainconfig.lr_eps
    lr_decay_rate = trainconfig.lr_decay_rate
    loss_eps = trainconfig.loss_eps
    try:
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join('.',file_to_save_model)))
        print("restore previous model")
        # create the file to save training & testing accuracy and loss
        curve = open(os.path.join('.',file_to_save_model,'loss_acc_curve.txt'),'a+')
        log = open(os.path.join('.',file_to_save_model,'train_log.txt'),'a+')
        curve.seek(0)
        lines = curve.readlines()
        lastline = lines[-1].strip('\n').split('\t')
        last_loss = float(lastline[-1])
        epoch_base = int(float(lastline[0])) + 1
        learning_rate = float(lastline[1])
        learning_rate = learning_rate * lr_decay_rate
        max_acc = float(lastline[2])
        log.write('[PARAMETER SETTING] initial_learning_rate: %f, lr_eps: %f, lr_decay_rate: %f, batch_size: %i, num_epoch: %i \n'%(learning_rate, eps, lr_decay_rate, batch_size, num_epoch))
        print('start from epoch: %i, learning rate: %f, last_loss: %f, max_acc: %f' %(epoch_base,learning_rate,last_loss,max_acc) )
    except:
        print("build new model")
        # create the file to save traing&testing accuracy and loss
        curve = open(os.path.join('.',file_to_save_model,'loss_acc_curve.txt'),'w')
        curve.write('epoch'+'\t'+'learning rate'+'\t'+'training accuracy'+'\t'+'testing accuracy'+'\t'+'loss'+'\n')
        log = open(os.path.join('.',file_to_save_model,'train_log.txt'),'w')
        learning_rate = trainconfig.initial_learning_rate
        max_acc = 0 # maximum accuracy, usage: to compare training accuracy at each epoch and save the best model
        last_loss = 0
        epoch_base = 0
        log.write('[PARAMETER SETTING] initial_learning_rate: %f, lr_eps: %f, lr_decay_rate: %f, batch_size: %i, num_epoch: %i \n'%(learning_rate, eps, lr_decay_rate, batch_size, num_epoch))
    LR_decay=True
    ###----- loading all data -----###
    X_test, y_test, _, _ = model.readdata(data_dir+'testing_dt.json')
    testinputs, testlabels, testseq_length = tagger.pre_process(X_test, y_test)
    
    X_train_ls=[]
    y_train_ls=[]
    train_input_ls=[]
    train_label_ls=[]
    train_seqlength_ls=[]

    for num in range(num_of_training_dt):
        #dataname = 'training_dt'+str(num)+'.json'
        dataname = 'training_dt.json'
        log.write('start to load '+dataname+'\n');log.flush()
        start_mini = timeit.default_timer()
        X_train, y_train, _, _ = model.readdata(data_dir+dataname)
        traininputs, trainlabels, trainseq_length = tagger.pre_process(X_train, y_train)
        X_train_ls.append(X_train)
        y_train_ls.append(y_train)
        train_input_ls.append(traininputs)
        train_label_ls.append(trainlabels)
        train_seqlength_ls.append(trainseq_length)
        log.write(dataname +' loaded. i/o time:'+str(timeit.default_timer()-start_mini)+' seconds'+'\n');log.flush()

    for epoch in range(epoch_base, num_epoch):
        start = timeit.default_timer()
        
        ###-----training step-----###
        for num in range(num_of_training_dt):
            start_mini = timeit.default_timer()
            X_train = X_train_ls[num]
            y_train = y_train_ls[num]
            #shuffle(X_train, y_train)   

            # split the data into minibatch
            [x_batch, y_batch] = model.multiminibatch([X_train, y_train], batch_size)
            #counter=0
            for x,y in zip(x_batch,y_batch):
                inputs, labels, seq_length = tagger.pre_process(x,y)
                #try:
                sess.run([tagger.train_op], feed_dict={
                    tagger.x: inputs, 
                    tagger.y: labels, 
                    tagger.length: seq_length, 
                    tagger.dropout: 0.2, 
                    tagger.lr: learning_rate})   
                #sess.run(tagger.x, feed_dict={tagger.x: inputs})
                #sess.run(tagger.y, feed_dict={tagger.y: labels})
                #sess.run(tagger.length, feed_dict={tagger.length: seq_length}) 
                #sess.run(tagger.dropout, feed_dict={tagger.dropout: 0.2})
                #sess.run(tagger.lr, feed_dict={tagger.lr: learning_rate})      
                #except:
                #	print(counter)
                #	print(x[0],y[0],seq_length)
                #	exit()
                #counter+=1
            log.write('epoch: '+str(epoch+1)+', training_dt'+str(num)+', train_op run time: '+str(timeit.default_timer()-start_mini)+' seconds'+'\n'); log.flush()

        
        ###-----calculate loss and training accuracy-----###
        for num in range(num_of_training_dt):
            start_mini = timeit.default_timer() 
            [inputs,labels,seq_length] = model.multiminibatch([train_input_ls[num], train_label_ls[num], train_seqlength_ls[num]], batch_size=1000)
            loss=[]; acc=[]
            for x,y,z in zip(inputs,labels,seq_length):
                loss += sess.run([tagger.loss], feed_dict={tagger.x: x, tagger.y: y, tagger.length: z, tagger.dropout: 0.0}) # return a list of one element
                acc += sess.run([tagger.accuracy], feed_dict={tagger.x: x, tagger.y: y, tagger.length: z, tagger.dropout: 0.0}) # return a list of one element
            log.write('epoch: '+str(epoch+1)+', training_dt'+str(num)+', calculating loss & acc run time: '+str(timeit.default_timer()-start_mini)+' seconds'+'\n'); log.flush()
        LOSS=sum(loss)/len(loss)
        ACC=sum(acc)/len(acc)
        
        log.write('epoch: '+str(epoch+1)+', loss: '+str(LOSS)+', training accuracy: '+str(ACC)+'\n'); log.flush()
        
        ###-----calculate testing accuracy-----###
        start_mini = timeit.default_timer() 
        test_acc=0
        [inputs,labels,seq_length] = model.multiminibatch([testinputs, testlabels, testseq_length], batch_size=1000)
        for x,y,z in zip(inputs,labels,seq_length):
            test_acc += sess.run(tagger.accuracy, feed_dict={tagger.x: x, tagger.y: y, tagger.length: z, tagger.dropout: 0.0})
        test_acc = test_acc/len(inputs)
        log.write('epoch: '+str(epoch+1)+', testing accuracy: '+str(test_acc)+', learing rate: '+str(learning_rate)+', run time: '+str(timeit.default_timer()-start_mini)+' seconds'+'\n'); log.flush()
        print('epoch: %d, loss: %f, learning rate: %f, training accuracy: %f, testing accuracy: %f, run time: %f seconds' %(epoch+1, LOSS, learning_rate, ACC, test_acc, timeit.default_timer()-start))

        ###-----write training information into loss_acc_curve.txt-----###
        curve.write(str(epoch+1)+'\t'+str(learning_rate)+'\t'+str(ACC)+'\t'+str(test_acc)+'\t'+str(LOSS)+'\n');curve.flush()
        
        ###-----learning rate decay-----###
        if learning_rate < 1e-6:
            LR_decay = False
        if LR_decay:
            if last_loss != 0 and abs(last_loss - LOSS)/last_loss < eps:
                learning_rate = learning_rate * lr_decay_rate
            last_loss = LOSS
        
        ###-----save better model-----###        
        if ACC > max_acc:
            max_acc = ACC
            saver.save(sess,os.path.join('.',file_to_save_model,'tagging-model.ckpt'))
            log.write('epoch: '+str(epoch+1)+', better model saved.'+'\n'); log.flush()

    log.close()
    curve.close()
    sess.close()

if __name__ == "__main__":
    #file_to_save_model = input("name the file to save the model: ")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", type=str, dest="data_dir", default='../en/',help="specify your data directory")
    parser.add_argument("--model_path", type=str, default='lstmcrf', dest="file_to_save_model",help="name the file to save your model")
    parser.add_argument("--model_type", type=str, default='3', dest="flag", help="the model to be trained \n1: LSTM-RNN \n2: BiLSTM-RNN \n3: LSTM+CRF \n4: BLSTM+CRF \n")
    arg=parser.parse_args()
    data_dir = arg.data_dir
    file_to_save_model = arg.file_to_save_model
    flag = arg.flag
    #data_dir = '../zh/'
    #data_dir = '../fr/'
    #data_dir = '../en/'
    #flag = input("please choose the model to be trained \n1: LSTM-RNN \n2: BiLSTM-RNN \n3: LSTM+CRF \n4: BLSTM+CRF \n")
    #train(flag, os.path.join('..',file_to_save_model), data_dir)
    train(flag, os.path.join(data_dir, file_to_save_model), data_dir)



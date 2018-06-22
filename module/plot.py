# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:25:06 2018

@author: wanting_huang
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def plotcurve(data):
    plt.figure(figsize=(10,8))
    plt.subplot(2, 1, 1)
    plt.plot(data['epoch'],data['loss'],'g-',label='loss')
    plt.title('POS tagging')
    plt.legend()
    plt.ylabel('cross-entropy (loss)')
    plt.subplot(2, 1, 2)
    plt.plot(data['epoch'],data['training accuracy'],'b-',label='train acc')
    plt.plot(data['epoch'],data['testing accuracy'],'r-',label='test acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('classification rate')
    plt.savefig("acc_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data_path", type=str, default='../en/', dest="data_dir", help="specify your data directory")
    parser.add_argument("--model_path", type=str, default='lstmcrf0621', dest="file_to_save_model",help="name the file to save your model")
    parser.add_argument("--model_type", type=str, default='3', dest="flag", help="the model to be trained \n1: LSTM-RNN \n2: BiLSTM-RNN \n3: LSTM+CRF \n4: BLSTM+CRF \n")
    arg=parser.parse_args()
    data_dir = arg.data_dir
    file_to_save_model = arg.file_to_save_model
    flag = arg.flag
    data = pd.read_table(os.path.join(data_dir,file_to_save_model,'loss_acc_curve.txt'))
    plotcurve(data)
'''
data['learning rate']
data.columns
data['training accuracy']
data['testing accuracy']

#plt.xlim(0,20)
plt.figure(figsize=(6,5))
plt.figure(figsize=(6,5))
plt.legend()
plt.legend()
plt.plot(data['epoch'],data['loss'],'b-',label='loss')
plt.plot(data['epoch'],data['testing accuracy'],'r-',label='test acc')
plt.plot(data['epoch'],data['training accuracy'],'b-',label='train acc')
plt.show()
plt.show()
plt.title('POStagging accuracy')
plt.title('POStagging loss')
plt.xlabel('epoch')
plt.xlabel('epoch')
plt.ylabel('classification rate')
plt.ylabel('cross-entropy')
plt.ylim(None,1.01)
'''


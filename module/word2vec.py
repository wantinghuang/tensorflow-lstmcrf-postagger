# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:22:36 2018

@author: wanting_huang
"""
# refer to http://zake7749.github.io/2016/08/28/word2vec-with-gensim/
# and https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c
import os
import logging
from gensim.models import word2vec
from gensim.models.fasttext import FastText
import json
#os.chdir('..')
#os.chdir('../fr')
os.chdir('../en')

###################################################
#####------- Word Embedding: word2vec ------#######
###################################################

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence("data_seg_train.txt")
model = word2vec.Word2Vec(sentences, size=100,min_count=1)

#保存模型，供日後使用
model.save("word2vec.model")
'''
model = word2vec.Word2Vec.load("word2vec.model")
'''
###################################################
#####------- Word Embedding: FastText ------#######
###################################################


sentences = word2vec.LineSentence("data_seg_train.txt")
model_ted = FastText(sentences, size=100, window=5, min_count=1, workers=4,sg=0,min_n=1,max_n=4)
model_ted.save("fasttext.model")
'''
model_ted = word2vec.Word2Vec.load("fasttext.model")
'''

####################################################
###---- function of transforming data to dict ---###
####################################################
## 把字轉換成向量儲存起來 / transform data from word to vector

with open('tagdt.json', 'r') as f:
    tagdt = json.load(f)

def datalabel2dt(data,label,save_name,embedding_size=100): 
    error_sen=[]
    blanksen=0
    dt={}
    print('data length:',len(data))
    for i in range(len(data)):
        sen = data[i]
        sen_lab = label[i]
        sen = sen.strip(' \n') # notice that there must be a space, or sen_ls will have one '' element and result in incorrect sentence length
        sen_lab = sen_lab.strip(' \n')
        if not sen.isspace() and sen!='': # ''.isspace() ==> False; ' '.isspace() ==>True
            sen_ls = sen.split(' ')
            sen_lab_ls = sen_lab.split(' ')
            if len(sen_ls) == len(sen_lab_ls):
                dt[i]={'word2vec':[],'onehot_label':[],
                        'fasttext':[],'length':len(sen_ls),
                        'original_sentence':sen,'original_label':sen_lab,
                        'origin':''}
                #for j in range(len(sen_ls)):
                #    word = sen_ls[j]
                #    tag = sen_lab_ls[j]
                for word,tag in zip( sen_ls, sen_lab_ls ):
                    if word!='' and tag!='':
                        dt[i]['origin'] += word+'_'+tag+' '
                        dt[i]['onehot_label'].append(tagdt[tag])
                        try:
                            dt[i]['word2vec'].append(model[word].tolist())
                        except:
                            dt[i]['word2vec'].append([0]*embedding_size)
                        try: 
                            dt[i]['fasttext'].append(model_ted[word].tolist())
                        except:
                            dt[i]['fasttext'].append([0]*embedding_size)
                    
                    elif word=='' and tag=='':
                        pass
                    else:
                        print("Fatal Error: word without tag or tag without word" , i, sen_ls, sen_lab_ls)
            else:
                print('Fatal Error2: len(sen_ls)!= len(sen_lab_ls)')
                return i, sen, sen_lab
        elif sen=='':
            blanksen+=1
        else:
            error_sen.append((i,sen))
            
    with open(save_name+'.json','w') as f:
        json.dump(dt,f)
    print(save_name+'.json done!')
    print('number of blank sentence:',blanksen)
    #print('problem sentence:',error_sen)
    return error_sen

####################################################
###----- transform data from word to vector -----###
####################################################

with open ('data_seg_train.txt','r',encoding='utf-8') as f:
    data = f.readlines()
    
with open('label_seg_train.txt','r',encoding='utf-8') as f:
    label=f.readlines()

datalabel2dt(data,label,save_name='training_dt',embedding_size=100)
'''
error_sen=[]
blanksen=0
step = int(len(data)/10)
for ix in range(10):
    dt = {}
    print(ix,ix*step,min((ix+1)*step, len(data)), 'data length:',min((ix+1)*step, len(data))-ix*step)
    for i in range(ix*step,min((ix+1)*step, len(data))):
        sen = data[i]
        sen_lab = label[i]
        sen=sen.strip(' \n')
        sen_lab=sen_lab.strip(' \n')
        if not sen.isspace() and sen!='': # ''.isspace() ==> False; ' '.isspace() ==>True
            sen_ls = sen.split(' ')
            sen_lab_ls = sen_lab.split(' ')
            if len(sen_ls)== len(sen_lab_ls):
                dt[i]={'word2vec':[],'onehot_label':[],
                        'fasttext':[],'length':len(sen_ls),
                        'original_sentence':sen,'original_label':sen_lab,
                        'origin':''}
                for word,tag in zip( sen_ls, sen_lab_ls ):
                    if word!='' and tag!='':
                        dt[i]['origin'] += word+'_'+tag+' '
                        dt[i]['onehot_label'].append(tagdt[tag])
                        try:
                            dt[i]['word2vec'].append(model[word].tolist())
                        except:
                            dt[i]['word2vec'].append([0]*embedding_size)
                        try:
                            dt[i]['fasttext'].append(model_ted[word].tolist())
                        except:
                            dt[i]['fasttext'].append([0]*embedding_size)
                    elif word=='' and tag=='':
                        pass
                    else:
                        print("Fatal Error: word without tag or tag without word" , i, sen_ls, sen_lab_ls)
            else:
                print('Fatal Error2: len(sen_ls)!= len(sen_lab_ls)')
                print(i, sen, sen_lab)
        elif sen=='':
            blanksen+=1
        else:
            error_sen.append((i,sen))
    with open('training_dt'+str(ix)+'.json','w') as f:
        json.dump(dt,f)
    print('training_dt'+str(ix)+'.json done')
    print('number of blank sentence:',blanksen)
    #print('problem sentence:',error_sen)
'''
'''
step = int(len(data)/10)
#print(step,len(data))
for ix in range(10):
    print(ix*step , (ix+1)*step)
    data = data[ix*step : (ix+1)*step]
    datalabel2dt(data,label,save_name='training_dt'+str(ix),embedding_size=100)
'''


###############################################
###------ save testing data into dict ------###
###############################################

with open ('data_seg_test.txt','r',encoding='utf-8') as f:
    data = f.readlines()

with open('label_seg_test.txt','r',encoding='utf-8') as f:
    label=f.readlines()

datalabel2dt(data,label,save_name='testing_dt',embedding_size=100)

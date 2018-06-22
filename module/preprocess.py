
import json
import os
#os.chdir('..')
os.chdir('../en')
'''
os.chdir('../fr')
### deal with corpus segmentation ###
#with open('zh.corpus','r', encoding='utf8') as f:
with open('fr.txt.tok.stanford-pos','r', encoding='utf8') as f:
    data = f.readlines()

###################################################
### --- create data_seg.txt & lebel_seg.txt --- ###
###################################################
output = open('data_seg.txt', 'w', encoding='utf-8') # extract word
label = open('lebel_seg.txt', 'w', encoding='utf-8') # extract label
count=0
for sen in data:
    words = sen.strip('\n').split(' ')
    for word in words:
        try:
            output.write('_'.join(word.split('_')[:-1]) + ' ')
            label.write(word.split('_')[-1].strip('\n') + ' ')
        except:
            print(count,'|',word,'.')
    output.write('\n')
    label.write('\n')
    count+=1

output.close()
label.close()
'''
'''
###################################################
### split corpus into training and testing data ###
###################################################
with open('data_seg.txt','r',encoding='utf-8') as f:
    data_seg = f.readlines()
#n=500000
#n=80000
n=50000
data_seg_train = data_seg[:n]
data_seg_test = data_seg[n:]

with open('data_seg_train.txt','w',encoding='utf-8') as f:
    for sen in data_seg_train:
        f.write(sen)
with open('data_seg_test.txt','w',encoding='utf-8') as f:
   for sen in data_seg_test:
        f.write(sen)

with open('lebel_seg.txt','r',encoding='utf-8') as f:
    label_seg = f.readlines()
label_seg_train = label_seg[:n]
label_seg_test = label_seg[n:]

with open('label_seg_train.txt','w',encoding='utf-8') as f:
    for sen in label_seg_train:
        f.write(sen)
with open('label_seg_test.txt','w',encoding='utf-8') as f:
    for sen in label_seg_test:
        f.write(sen)
'''
##################################
####---- label dictionary ----####
##################################
'''
### deal with pos label ###
with open('zh.PoS','r', encoding='utf8') as f:
    pos = f.readlines()

# create a pos dict
posdt={}
for i in range(len(pos)):
    posdt[pos[i].strip('\n')]=i
posdt.values()


tag_dt={}
for key in posdt.keys():
    vec = [0]*len(pos)
    vec[posdt[key]] = 1
    tag_dt[key]=vec

with open('tagdt.json', 'w') as f:
    json.dump(tag_dt,f)
    
with open('posdt.json', 'w') as f:
    json.dump(posdt,f)
'''

with open('label_seg_train.txt','r',encoding='utf-8') as f:
    data = f.readlines()
with open('label_seg_test.txt','r',encoding='utf-8') as f:
    data2 = f.readlines()
posls=set()
for sen in data:
    sen = sen.strip(' \n')
    wordls = sen.split(' ')
    for word in wordls:
        posls.add(word)
for sen in data2:
    sen = sen.strip(' \n')
    wordls = sen.split(' ')
    for word in wordls:
        posls.add(word)
print(len(posls),posls)

# create a pos dict
posdt={}
count=0
for tag in posls:
    posdt[tag]=count
    count+=1
posdt.values()

tag_dt={}
for key in posdt.keys():
    vec = [0]*len(posls)
    vec[posdt[key]] = 1
    tag_dt[key]=vec
    
with open('tagdt.json', 'w') as f:
    json.dump(tag_dt,f)
    
with open('posdt.json', 'w') as f:
    json.dump(posdt,f)

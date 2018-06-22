from nltk.corpus import brown
import nltk
import pickle
import time
## Reference
# https://www.nltk.org/book/ch02.html
# https://www.nltk.org/book/ch05.html
# https://blog.csdn.net/fxjtoday/article/details/5841453
# https://blog.csdn.net/zhuzuwei/article/details/79008816

brown.categories()
len(brown.categories())
## combine brown corpus
corpus=[]
for genre in brown.categories():
    print(genre, len(brown.tagged_sents(categories=genre)))
    corpus += brown.tagged_sents(categories=genre)
len(corpus)
corpus[40]
       

## record all seen tag(brown corpus taglist)
tag = set()
for sen in corpus:
    for wordset in sen:
        tag.add(wordset[1])
len(tag)

f=open("brown_taglist.txt","w")
for each in tag:
    f.write(each+'\n')
f.close()


## create eng_seg.txt and label_seg.txt
f = open('C:\\Users\\wanting_huang\\Desktop\\tagging\\data\\eng_seg.txt','w',encoding='utf-8')
g = open('C:\\Users\\wanting_huang\\Desktop\\tagging\\data\\label_seg.txt','w',encoding='utf-8')

for sen in corpus:
    for wordset in sen:
        f.write(wordset[0]+' ')
        g.write(wordset[1]+' ')
    f.write('\n')
    g.write('\n')
f.close()
g.close()


## NLTK n-gram tagger
train_sents = corpus[:50000]
test_sents = corpus[50000:]
start=time.time()
t0 = nltk.DefaultTagger('NN')  
t1 = nltk.UnigramTagger(train_sents,backoff=t0)  
t2 = nltk.BigramTagger(train_sents,backoff=t1)  
print('run time:', time.time()-start)
print('training accuracy', t2.evaluate(train_sents))
print('testing accuracy', t2.evaluate(test_sents))

## see tagging result 
# use word_tokenize to segment sentence
text = nltk.word_tokenize("And now for something completely different...")
# check our own trained tagger
t2.tag(text)
# compare to nltk.pos_tag
nltk.pos_tag(text)


## see tag information
# nltk.download('tagsets')
with open('C:\\Users\\wanting_huang\\AppData\\Roaming\\nltk_data\\help\\tagsets\\brown_tagset.pickle','rb') as f:
    ls = pickle.load(f)
ls.keys()

f=open('brown_tagsets.txt','w',encoding='utf-8')
for key, value in ls.items():
    f.write(key+' // '+value[0]+ '. E.g., ' + value[1] + '\n')
f.close()

## see those who are in the corpus but not in brown_tagsets.txt
unknowntag = set()
for sen in corpus:
    for wordset in sen:
        if wordset[1] not in ls:
            unknowntag.add(wordset[1])
print(unknowntag  )


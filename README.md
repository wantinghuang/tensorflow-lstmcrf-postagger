# Tackling the POS tagging problem with Neural Network model 

Part-of-Speech (POS) tagging problem has been studied in the field of computational lingustics for several years. N-gram tagger[1] is the most popular POS tagger in the past. As neural network model growing rapidly, let's consider training a POS tagger with neural network model.

## Dependencies
* python 3
* modules
    * TensorFlow 1.8.0
    * json
    * os
    * numpy
    * argparse
    * string

## Usage
Type with such order in command line
```
> python preprocess.py
> python word2vec.py
> python train.py
> python predict.py
> python argv.py
```

## Sample Usage of argv.py 
(after go through all steps above)
```
> python argv.py
write down sentences and see the POS tags: (exit: enter "q")
> yeah! it's friday. Let's celebrate tonight!
yeah_NN !_. it_PPS 's_PPS+BEZ friday_JJ ._. Let_NN-TL-HL 's_NP$-HL celebrate_JJ tonight_NN !_.

```

## Overview

### Goal
train a POS tagger with neural network model

### Dataset
NLTK Brown Corpora

There are 15 categories in Brown corpora[2], with 57340 english sentences and 472 tags.
We use the former 50000 sentences as training data and the rest as testing data.

### Preprocess Data
Create 
1. two files that seperate the word and its POS tag.
2. word embedding usnig word2vec and fasttext using Gensim[3]. embedding size = 100。
3. one hot encoding for POS tag 

### Modelling
Use LSTM, BLSTM, LSTM+CRF, BLSTM+CRF models
** BLSTM is short for bi-directional LSTM **

### Evaluation
per token accuracy


## Development Process

preprocess data(1,2) --> model(3) --> trian(4) --> check learning curve & overfitting(7) --> check output label(5) --> interactive demo(6)

* numbers in () refer to code number, see APPENDIX


## Parameter Setting
1. Modeling Step
    * embedding_size = 100
    * number_of_tags = 472
    * neurons/nodes_in_hidden_layer = 128
    * max_time_step = 105 # number of words in the longest sentence
2. Training Step
    * batch_size = 128 # should it be larger or smaller?
    * number_of_epoch = 25  # should set stopping criteria
    * initial_learning_rate = 0.001
    * lr_decay_rate = 0.1 # using decay learning rate
    * lr_eps = 1e-3 # when abs(last_loss - current_loss)/last_loss < lr_eps, learning rate decay 
    (ie. lr <- lr_decay_rate * lr)


## Experimental Results

#### DEMO
* example 1
```
[true] He_PPS does_DOZ not_* ._. 
[pred] He_PPS does_DOZ not_* ._.
```    
* example 2
```
[true] But_CC very_QL mystical_JJ too_RB ._. 
[pred] But_CC very_QL mystical_JJ too_NN ._. 
```


#### Benchmarking

compare LSTM, BLSTM, LSTM+CRF, BLSTM+CRF model and NLTK n-gram tagger[4]


|   model   | training accuracy | testing accuracy | training time (sec) per epoch|
| --------- | ----------------- | ---------------- | ---------------------------- |
|LSTM       |     0.90          |      0.89        |          230                 |
|BLSTM      |     0.90          |      0.87        |          350                 |
|LSTM+CRF   |     0.94          |      0.91        |         1050                 |
|BLSTM+CRF  |     0.93          |      0.90        |         1110                 |
|NLTK n-gram|     0.96          |      0.90        |         11(total)            |


#### LOSS & ACCURACY CURVE
![lstmcrf_curve](https://i.imgur.com/qkJiwaa.png)


## Reference

[1] N-gram tagger is a "lookup tagger", which based on a simple statistical algorithm: for each token, assign the tag that is most likely for that particular token.

[2] [NLTK Category]( https://www.nltk.org/book/ch02.html )

[3] [Word2Vec and FastText Word Embedding with Gensim]( https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c )

[4] [NLTK POS tagging]( https://www.nltk.org/book/ch05.html )

[5] [POS Tagging (State of the art)]( https://aclweb.org/aclwiki/POS_Tagging_(State_of_the_art) )



## APPENDIX

### HOW THESE CODES WORK
1. preprocess.py: Process original data to data_seg.txt and lebel_seg.txt. Also, split the training and texting data.

2. word2vec.py: Train word embedding vector (fasttext & word2vec). Turn data_seg_train.txt + lebel_seg_train.txt into training_dt.json and testing_dt.json

3. model
3.1 parameter.py: hyperparameter setting
3.2 model.py: LSTM model
3.3 model_biRNN.py: BLSTM model
3.4 model_LSTMCRF.py: LSTM+CRF model
3.5 model_BLSTMCRF.py: BLSTM+CRF model

4. train.py: Train the model

5. predict.py: Output the comparison of predicted and true label

6. argv.py: Write down any sentence and return POS tags of each word in command line (interactive)

7. plot.py: Plot learning curve using matplotlib (NEED GUI)

### NLTK n-gram tagger code
```
# NLTK POStagger API
import nltk
t0 = nltk.DefaultTagger('NN')  # Default taggers 'NN' assign their tag to every single word
t1 = nltk.UnigramTagger(train_sents,backoff=t0)
t2 = nltk.BigramTagger(train_sents,backoff=t1)  
print('training accuracy', t2.evaluate(train_sents))
print('testing accuracy', t2.evaluate(test_sents))
```


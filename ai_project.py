#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Importing libraries
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time
text = open(r'C:\Users\asus\Desktop\train.txt','r').read()
text=text.split('. . O')


# In[22]:


data=[]
unwanted_tags=['#',
 '$',
 "''",
 '(',
 ')',
 ',',
 '.',
 ':']
for i in text:
    train_x=[]
    T=list(i.split('\n'))
    for j in T:
        
        m=j.split()
        if len(m)<=1 :
            continue
        elif m[1] in unwanted_tags:
            continue
        train_x.append((m[0],m[1]))
    data.append(train_x)
        
            
    


# In[23]:


# split data into training and validation set in the ratio 95:5
random.seed(1234)
train_set, test_set = train_test_split(data, train_size=0.95, test_size=0.05)

print("Training Set Length -", len(train_set))
print("Testing Set Length -", len(test_set))
print("-" * 100)
print("Training Data -\n")
print(train_set[:10])


# In[24]:


# Getting list of train and test tagged words
train_tagged_words = [tup for sent in train_set for tup in sent]
print("Train Tagged Words - ", len(train_tagged_words))

test_tagged_words = [tup[0] for sent in test_set for tup in sent]
print("test Tagged Words - ", len(test_tagged_words))


# In[25]:


# Let's have a look at the tagged words in the training set
train_tagged_words[:10]


# In[26]:


# tokens in the train set - train_tagged_words
train_tagged_tokens = [tag[0] for tag in train_tagged_words]
train_tagged_tokens[:10]


# In[27]:


# POS tags for the tokens in the train set - train_tagged_words

train_tagged_pos_tokens = [tag[1] for tag in train_tagged_words]
set(sorted(train_tagged_pos_tokens))


# In[28]:


# building the train vocabulary to a set
training_vocabulary_set = set(train_tagged_tokens)


# In[29]:


# building the POS tags to a set
training_pos_tag_set = set(train_tagged_pos_tokens)


# In[30]:


#tags are present in training data
print(len(training_pos_tag_set))


# In[31]:


# let's check how many words are present in vocabulary
print(len(training_vocabulary_set))


# In[32]:


# compute emission probability for a given word for a given tag
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    tag_count = len(tag_list)    
    word_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]    
    word_given_tag_count = len(word_given_tag_list)    
    
    return (word_given_tag_count, tag_count)


# In[33]:


# compute transition probabilities of a previous and next tag
def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    
    t1_tags_list = [tag for tag in tags if tag == t1]
    t1_tags_count = len(t1_tags_list)
    
    t2_given_t1_list = [tags[index+1] for index in range(len(tags)-1) if tags[index] == t1 and tags[index+1] == t2]
    t2_given_t1_count = len(t2_given_t1_list)
    
    return(t2_given_t1_count, t1_tags_count)


# In[34]:


# computing P(w/t) and storing in [Tags x Vocabulary] matrix. This is a matrix with dimension
# of len(training_pos_tag_set) X en(training_vocabulary_set)

len_pos_tags = len(training_pos_tag_set)
len_vocab = len(training_vocabulary_set)


# In[35]:


# creating t x t transition matrix of training_pos_tag_set
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)

tags_matrix = np.zeros((len_pos_tags, len_pos_tags), dtype='float32')
for i, t1 in enumerate(list(training_pos_tag_set)):
    for j, t2 in enumerate(list(training_pos_tag_set)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]


# In[36]:


# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(training_pos_tag_set), index=list(training_pos_tag_set))


# In[37]:


# Let's have a glimpse into the transition matrix
tags_df


# In[38]:


def Vanilla_Viterbi(words, train_bag = train_tagged_words):
    state = []
    
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))


# In[28]:


# Let's test our Viterbi algorithm on a few sample sentences of test dataset

random.seed(1234)

# choose random 5 sents
rndom = [random.randint(1, len(test_set)) for x in range(5)]

# list of sents
test_run = [test_set[i] for i in rndom]

# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]

# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]


# In[29]:


# tagging the test sentences
start = time.time()
tagged_seq = Vanilla_Viterbi(test_tagged_words)
end = time.time()
difference = end-start

print("Time taken in seconds: ", difference)

# accuracy
vanilla_viterbi_word_check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
vanilla_viterbi_accuracy = len(vanilla_viterbi_word_check)/len(tagged_seq) * 100
print('Vanilla Viterbi Algorithm Accuracy: ', vanilla_viterbi_accuracy)


# In[30]:


# let's check the incorrectly tagged words
incorrect_tagged_words = [j for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0] != j[1]]

print("Total Incorrect Tagged Words :", len(incorrect_tagged_words))
print("\n")
print("Incorrect Tagged Words :", incorrect_tagged_words)


# In[31]:


# Unknown words 

test_vocabulary_set = set([t for t in test_tagged_words])

unknown_words = list(test_vocabulary_set - training_vocabulary_set)
print("Total Unknown words :", len(unknown_words))
print("\n")
print("Unknown Words :", unknown_words)


# In[ ]:





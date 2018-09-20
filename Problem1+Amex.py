
# coding: utf-8

# # LDA topic modelling and similarity

# make a list of all test articles filenames

# In[1]:


from os import listdir
from os.path import isfile, join
import pandas as pd
mypath_w="D:/Competitions/AmEx/Dataset/test-data/"
onlyfiles_ew = [f for f in listdir(mypath_w) if isfile(join(mypath_w, f))]


# Loading train data

# In[2]:


import os
train_set=pd.DataFrame()
mypath_w="D:/Competitions/AmEx/Dataset/training-data/"
onlyfiles_ew = [f for f in listdir(mypath_w) if isfile(join(mypath_w, f))]
train_set['file_name']=onlyfiles_ew
os.chdir(mypath_w)
articles_train=[]
for fname in onlyfiles_ew:
    with open(fname) as infile:
        articles_train.append(infile.read())
train_set['articles']=articles_train
train_sol=pd.read_csv("D:/Competitions/AmEx/Dataset/train_data_segments.csv")
train_final=pd.merge(train_set,train_sol,how='left',on='file_name')
train_final["articles_new"]=train_final["articles"].str.split('-----')


# In[3]:


train_final.isna().sum()


# In[104]:


train_final.head()


# Dividing the train articles by topics and make an LDA data set for topic modelling

# In[4]:


slist =[]
for x in train_final.articles_new:
    slist.extend(x)


# In[5]:


train_for_lda=pd.DataFrame()
train_for_lda['articles']=slist


# Loading Validation and Test Data

# In[6]:


validation_set=pd.DataFrame()
mypath_w="D:/Competitions/AmEx/Dataset/val-data/"
onlyfiles_ew = [f for f in listdir(mypath_w) if isfile(join(mypath_w, f))]
b=onlyfiles_ew.index('.DS_Store')
del onlyfiles_ew[b]
validation_set['file_name']=onlyfiles_ew
os.chdir(mypath_w)
articles_valid=[]
for fname in onlyfiles_ew:
    with open(fname) as infile:
        articles_valid.append(infile.read())
validation_set['articles']=articles_valid
validation_sol=pd.read_csv("D:/Competitions/AmEx/Dataset/val_data_segments.csv")
val_final=pd.merge(validation_set,validation_sol,how='left',on='file_name')
val_final.isna().sum()


# In[105]:


val_final.head()


# In[106]:


val_final["articles_new"]=val_final["articles"].str.split('-----')


# In[110]:


len(val_final)


# In[111]:


len(train_final)


# In[107]:


for x in val_final.articles_new:
    slist.extend(x)


# In[108]:


train_for_lda=pd.DataFrame()
train_for_lda['articles']=slist


# In[109]:


len(train_for_lda)


# In[122]:


test_set=pd.DataFrame()
mypath_w="D:/Competitions/AmEx/Dataset/test-data/"
onlyfiles_ew = [f for f in listdir(mypath_w) if isfile(join(mypath_w, f))]
#b=onlyfiles_ew.index('.DS_Store')
#del onlyfiles_ew[b]
test_set['file_name']=onlyfiles_ew
os.chdir(mypath_w)
articles_test=[]
for fname in onlyfiles_ew:
    with open(fname) as infile:
        articles_test.append(infile.read())
test_set['articles']=articles_test
#validation_sol=pd.read_csv("D:/Competitions/AmEx/Dataset/val_data_segments.csv")
#val_final=pd.merge(validation_set,validation_sol,how='left',on='file_name')
test_set.isna().sum()


# In[8]:


del validation_sol,validation_set,train_sol,train_set


# Splitting the test set by fullstop sign and counting the number of splits

# In[9]:


import gc
gc.collect()


# In[123]:


test_set["articles_new"]=test_set["articles"].str.split('.')


# In[124]:


test_set['list_cnt']=0
for i in range(len(test_set)):
    test_set['list_cnt'][i]=len(test_set['articles_new'][i])


# In[125]:


slist_test =[]
for x in test_set.articles_new:
    slist_test.extend(x)


# Creating the index column of filenames for the split dataset

# In[126]:


list_index=[]
for i in range(len(test_set)):
    for j in range(test_set['list_cnt'][i]):
        list_index.append(test_set['file_name'][i])


# In[127]:


test_for_lda=pd.DataFrame()
test_for_lda['articles']=slist_test
test_for_lda['filename']=list_index
test_for_lda.head()


# Now that the data preparation is done we do data cleaning using nltk package

# In[15]:


import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


# In[16]:


def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

stop_words = stopwords.words('english')
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]

stemmer = PorterStemmer()
def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))


# In[112]:


train_for_lda["articles_clean"]=""
for i in range(len(train_for_lda)):
    train_for_lda["articles_clean"][i]=apply_all(train_for_lda["articles"][i])
    print(i)


# In[18]:


test_for_lda=test_for_lda.reset_index()


# In[128]:


test_for_lda["articles_clean"]=""
for i in range(len(test_for_lda)):
    test_for_lda["articles_clean"][i]=apply_all(test_for_lda["articles"][i])
    print(i)


# In[129]:


test_for_lda.head()


# In[21]:


test_for_lda=test_for_lda.drop('index',axis=1)


# Looking at word frequency

# In[130]:


word_list=[]
for x in train_for_lda.articles_clean:
    word_list.extend(x)
word_list_dup=word_list
from collections import Counter
word_list_series=Counter(word_list_dup)


# Training the LDA model

# In[191]:


def train_lda(data):
    """
    This function trains the lda model
    We setup parameters like number of topics, the chunksize to use in Hoffman method
    We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
    """
    num_topics = 100
    chunksize = 300
    dictionary = corpora.Dictionary(data['articles_clean'])
    corpus = [dictionary.doc2bow(doc) for doc in data['articles_clean']]
    t1 = time.time()
    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-3, eta=0.5e-3, chunksize=chunksize, minimum_probability=0.0, passes=3)
    t2 = time.time()
    print("Time to train LDA model on ", len(data), "articles: ", (t2-t1)/60, "min")
    return dictionary,corpus,lda


# In[192]:


dictionary = corpora.Dictionary(train_for_lda['articles_clean'])


# In[194]:


dictionary,corpus,lda = train_lda(train_for_lda)


# In[158]:


#see top 10 topics
lda.show_topics(num_topics=10, num_words=20)


# In[159]:


#saving the model
from gensim.test.utils import datapath
fname = datapath("D:/Competitions/AmEx/Dataset-PB1/model_30")
lda.save(fname)


# In[22]:


from gensim.test.utils import datapath
fname = datapath("D:/Competitions/AmEx/Dataset-PB1/model_100_1")
lda = LdaModel.load(fname, mmap='r')


# In[131]:


test_for_lda['articles_len']=test_for_lda['articles'].str.len()


# Considering the articles which only have more than 40 characters for topic modelling and creating a copy of the original test set

# In[132]:


test_for_lda_copy=test_for_lda
test_for_lda=test_for_lda.loc[test_for_lda['articles_len']>40]


# In[134]:


len(test_for_lda_copy)


# In[135]:


test_for_lda_list=list(test_for_lda.articles_clean)


# In[172]:


#other_corpus


# Getting topic loading for the test data for the calculation of the topic similarity

# In[200]:


import numpy as np
other_corpus = [dictionary.doc2bow(text) for text in test_for_lda_list]
unseen_doc = other_corpus[0]
vector = lda[unseen_doc]
columns=[tup[0] for tup in lda.get_document_topics(bow=vector)]
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=vector)])
similarity=pd.DataFrame(new_doc_distribution.reshape(-1, len(new_doc_distribution)),columns=columns)
for i in range(len(other_corpus)):
    unseen_doc = other_corpus[int(i+1)]
    vector = lda[unseen_doc]
    new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=vector)])
    similarity.loc[int(i+1)]=new_doc_distribution
    print(i)


# In[201]:


similarity.head()


# In[137]:


list_names=list(test_for_lda.filename.unique())


# Calculating the similarity

# In[138]:


names=test_for_lda[['filename']].reset_index()
similarity_1=pd.concat([names,similarity],axis=1)


# In[139]:


similarity_result=dict(tuple(similarity_1.groupby('filename')))


# In[43]:


#similarity_result[list_names[0]].iloc[i,2:]


# Calculating Cosine similarity of the topic loadings for different sentences

# In[140]:


sim=[]
from scipy import spatial
for f in list_names:
    print(f)
    for i in range(int(len(similarity_result[f])-1)):
        #print(i)
        result = spatial.distance.cosine(list(similarity_result[f].iloc[i,2:]), list(similarity_result[f].iloc[i+1,2:]))
        sim.append(result)
    sim.append(int(0))


# In[141]:


test_for_lda['cosine_sim']=sim


# In[142]:


get_ipython().magic('matplotlib inline')
plt.hist(test_for_lda.cosine_sim,bins=20)


# In[143]:


test_for_lda['rank']=test_for_lda.groupby('filename')['cosine_sim'].rank(method='dense').astype(int)


# In[144]:


test_for_lda=test_for_lda.reset_index()
test_for_lda=test_for_lda.drop('index',axis=1)


# In[145]:


#no of splits we want per topic
test_for_lda['flag']=0
for i in range(len(test_for_lda)):
    if test_for_lda['rank'][i]<5:
        test_for_lda['flag'][i]=1
    print(i)


# In[150]:


test_for_lda_copy=pd.merge(test_for_lda_copy,test_for_lda[['articles','flag']],on='articles',how='left')
test_for_lda_copy=test_for_lda_copy.fillna(0)


# In[151]:


test_for_lda.head()


# In[148]:


test_for_lda_copy=test_for_lda_copy.drop(['flag'],axis=1)


# In[152]:


man_csv=test_for_lda_copy[['filename','flag','articles_len']]


# In[153]:


man_csv.head(20)


# In[154]:


man_csv=man_csv.loc[man_csv.articles_len>0]
man_csv['flag_shifted']=man_csv.groupby(['filename'])['flag'].shift(1)
man_csv=man_csv.fillna(0)
man_csv['cumulative'] = man_csv.groupby(['filename'])['flag_shifted'].apply(lambda x: x.cumsum())


# In[198]:


man_csv['cumulative_article_len'] = man_csv.groupby(['filename'])['articles_len'].apply(lambda x: x.cumsum())


# In[199]:


sub=man_csv[['filename','cumulative','cumulative_article_len']].groupby(['filename','cumulative']).max()
sub=pd.DataFrame(sub.to_records())
sub_final=sub.groupby('filename')['cumulative_article_len'].apply(list)
sub_final=pd.DataFrame(sub_final)
sub_final=pd.DataFrame(sub_final.to_records())
sub_final.columns=['file_name','segments']
sub_final.to_csv('D:/Competitions/AmEx/Dataset-PB1/sub_final_cosine_val_cumm.csv',index=False)


# In[93]:


#sub_final.to_csv('D:/Competitions/AmEx/Dataset-PB1/sub_final_cosine_250.csv',index=False)


# Testing for euclidean distance assuming LDA give normal topic distributions with scipy.spatial.distance.euclidean(u, v)

# In[202]:


sim_euclidean=[]
from scipy import spatial
for f in list_names:
    print(f)
    for i in range(int(len(similarity_result[f])-1)):
        #print(i)
        result = spatial.distance.euclidean(list(similarity_result[f].iloc[i,2:]), list(similarity_result[f].iloc[i+1,2:]))
        sim_euclidean.append(result)
    sim_euclidean.append(int(0))


# In[203]:


test_for_lda['euclidean_sim']=sim_euclidean


# In[204]:


get_ipython().magic('matplotlib inline')
plt.hist(test_for_lda.euclidean_sim,bins=20)


# In[205]:


test_for_lda['rank_euclidean']=test_for_lda.groupby('filename')['euclidean_sim'].rank(method='dense').astype(int)


# In[206]:


test_for_lda=test_for_lda.reset_index()
test_for_lda=test_for_lda.drop('index',axis=1)


# In[207]:


#no of splits we want per topic
test_for_lda['flag_euclidean']=0
for i in range(len(test_for_lda)):
    if test_for_lda['rank_euclidean'][i]<5:
        test_for_lda['flag_euclidean'][i]=1
    print(i)


# In[209]:


#test_for_lda_copy=test_for_lda_copy.drop('flag_euclidean',axis=1)


# In[210]:


test_for_lda_copy=pd.merge(test_for_lda_copy,test_for_lda[['articles','flag_euclidean']],on='articles',how='left')
test_for_lda_copy=test_for_lda_copy.fillna(0)


# In[211]:


man_csv=test_for_lda_copy[['filename','flag_euclidean','articles_len']]


# In[212]:


man_csv=man_csv.loc[man_csv.articles_len>0]
man_csv['flag_euclidean_shifted']=man_csv.groupby(['filename'])['flag_euclidean'].shift(1)
man_csv=man_csv.fillna(0)
man_csv['cumulative'] = man_csv.groupby(['filename'])['flag_euclidean_shifted'].apply(lambda x: x.cumsum())


# In[213]:


man_csv['cumulative_article_len'] = man_csv.groupby(['filename'])['articles_len'].apply(lambda x: x.cumsum())


# In[214]:


sub=man_csv[['filename','cumulative','cumulative_article_len']].groupby(['filename','cumulative']).max()
sub=pd.DataFrame(sub.to_records())
sub_final=sub.groupby('filename')['cumulative_article_len'].apply(list)
sub_final=pd.DataFrame(sub_final)
sub_final=pd.DataFrame(sub_final.to_records())
sub_final.to_csv('D:/Competitions/AmEx/Dataset-PB1/sub_final_euclidean_finalcummulative.csv',index=False)


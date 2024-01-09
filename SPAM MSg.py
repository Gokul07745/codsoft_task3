#!/usr/bin/env python
# coding: utf-8

# In[77]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk


# In[78]:


pip install chardet


# In[79]:


import chardet

with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read())

ds = pd.read_csv('spam.csv', encoding=result['encoding'])


# In[80]:


ds.head()


# In[81]:


print(result)


# In[82]:


ds.info()


# In[83]:


ds = ds.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[84]:


print(ds)


# In[85]:


ds.v2.sample(n=10)#sampling


# In[86]:


def remSpecCharac():
    mesms=[]
    for msg in ds['v2']:
        e_string= ''
        for char in msg.strip():
            if char == ' ' or char.isalnum() == True:
                e_string+=char
        mesms.append(e_string)
    return mesms


# In[87]:


cleansms= remSpecCharac()
ds["cleaned_sms"]=cleansms
ds


# In[88]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize   #tokenization


# In[89]:


sw= stopwords.words('english')
word =[] 
for sentences in ds['cleaned_sms']:
    wordtokens = word_tokenize(sentences)
    aftremovstpwrds = [words for words in wordtokens if words not in sw]
    word.append(" ".join(aftremovstpwrds))
word[:5]


# In[90]:


from nltk.stem import PorterStemmer


# In[91]:


def stem():# stemming
    from nltk.stem import PorterStemmer
    pr = PorterStemmer()
    sc = []
    for sentences in word:
        stemmedtokens = [pr.stem(words.lower())for words in word_tokenize(sentences)]
        sc.append(" ".join(stemmedtokens))
    return sc
sc = stem()
sc[:5]
     


# In[92]:


ds['stemedsms']= sc
ds


# In[95]:


x_train = ds.stemedsms.iloc[:5000].values
y_train = ds.v1.iloc[:5000].values
x_test =  ds.stemedsms.iloc[5000:].values
y_test = ds.v1.iloc[5000:].values
  


# In[96]:


from sklearn.feature_extraction.text import TfidfVectorizer



# In[97]:


vectorizer = TfidfVectorizer()
trainingVectors = vectorizer.fit_transform(x_train)
testingVectors = vectorizer.transform(x_test)

testingVectors.shape


# In[98]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(trainingVectors, y_train)


# In[99]:


from  sklearn.metrics  import accuracy_score,f1_score
predictedvalue  = clf.predict(testingVectors)


# In[101]:


print(f'Accuracy: {accuracy_score(y_test,predictedvalue)}')
print(f'F1 Score: {f1_score(y_test,predictedvalue,average="macro")}')


# In[ ]:





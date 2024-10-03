#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection - Explore with IRFAN

# ## Data frames

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('creditcard.csv')


# In[3]:


df


# In[4]:


df['Class'].value_counts()


# ### Data Pre-processing

# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


scalar = StandardScaler()


# In[8]:


X = df.drop('Class', axis=1)
y = df.Class


# In[9]:


X = scalar.fit_transform(X)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ## Modeling

# In[11]:


from sklearn.svm import SVC


# In[12]:


model_svc = SVC()


# In[13]:


model_svc.fit(X_train, y_train)


# In[14]:


model_svc.score(X_train,y_train)


# In[15]:


model_svc.score(X_test,y_test)


# In[16]:


y_predict = model_svc.predict(X_test)


# ## Implementing Report

# In[17]:


from sklearn.metrics import classification_report , confusion_matrix


# In[18]:


import numpy as np


# In[19]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
confusion


# In[20]:


import seaborn as sns


# In[21]:


sns.heatmap(confusion, annot=True)


# In[22]:


print(classification_report(y_test, y_predict))


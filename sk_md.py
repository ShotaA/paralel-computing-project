#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
import time


# In[23]:


filename='blood.csv'
#filename='2dplanes.csv'

n=20

data=pd.read_csv(filename)
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
#f=int(math.log(X.shape[1]+1,2))


# In[24]:


start=time.time()
clf = RandomForestClassifier(n_estimators=n,random_state=0)
clf.fit(X, y)
taken=time.time()-start
print(taken)


# In[25]:


start=time.time()
clf = BaggingClassifier(n_estimators=n,random_state=0)
clf.fit(X, y)
taken=time.time()-start
print(taken)


# In[ ]:





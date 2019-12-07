#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

sc = SparkContext(appName="mlModels")


# Helper Functions

# In[11]:


# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])
#Evaluate
def evaluate(model,trainingData,testData):
    predictions = model.predict(trainingData.map(lambda x: x.features))
    labelsAndPredictions = trainingData.map(lambda lp: lp.label).zip(predictions)
    trainErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(trainingData.count())
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(testData.count())
    return (trainErr,testErr)


# Process Data

# In[9]:


data = sc.textFile("blood_data.txt")
parsedData = data.map(parsePoint)
(trainingData, testData) = parsedData.randomSplit([0.8, 0.2])


# SVM with SGD

# In[12]:


from pyspark.mllib.classification import SVMWithSGD, SVMModel
model = SVMWithSGD.train(trainingData, iterations=100)
evaluate(model,trainingData,testData)


# Gradient Boosted Trees

# In[13]:


from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
model = GradientBoostedTrees.trainClassifier(parsedData,
                                             categoricalFeaturesInfo={}, numIterations=3)
evaluate(model,trainingData,testData)


# In[14]:


dat=evaluate(model,trainingData,testData)


# In[18]:


import pandas as pd
pd.DataFrame(dat).to_csv('resutls.txt',index=False,header=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
import time

sc = SparkContext(appName="mlModels")


# Helper Functions

# In[2]:


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

# In[3]:


#data = sc.textFile("blood_data.txt")
filename="blood_data.txt"

data = sc.textFile(filename)

parsedData = data.map(parsePoint)
(trainingData, testData) = parsedData.randomSplit([0.8, 0.2])


# SVM with SGD

# from pyspark.mllib.classification import SVMWithSGD, SVMModel
# start=time.time()
# model = SVMWithSGD.train(trainingData, iterations=100)
# print(time.time()-start)
# evaluate(model,trainingData,testData)

# Decisiont Tree

# from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
# start=time.time()
# model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
#                                      impurity='gini', maxDepth=5, maxBins=32)
# print(time.time()-start)
# evaluate(model,trainingData,testData)

# Random Forest

<<<<<<< HEAD
# from pyspark.mllib.tree import RandomForest, RandomForestModel
# model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
#                                      numTrees=3, featureSubsetStrategy="auto",
#                                      impurity='gini', maxDepth=4, maxBins=32)
# evaluate(model,trainingData,testData)

# In[16]:


from pyspark.mllib.tree import RandomForest, RandomForestModel
for n in [1,2,4,8,16,32,64,100]:
    start=time.time()
    model = RandomForest.trainClassifier(parsedData, numClasses=2, categoricalFeaturesInfo={},
                                         numTrees=n, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=30, maxBins=32)
    taken=time.time()-start
    taken
    pd.DataFrame([[filename,taken,n]]).to_csv('rf_spark.txt',mode='a',index=False,header=False)
=======
#from pyspark.mllib.classification import SVMWithSGD, SVMModel
#model = SVMWithSGD.train(trainingData, iterations=100)
#evaluate(model,trainingData,testData)
>>>>>>> 5d0356ada16b21acb718a3437da55805e6b554ef


# Gradient Boosted Trees

# In[6]:


#from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
#model = GradientBoostedTrees.trainClassifier(parsedData,
<<<<<<< HEAD
                                             categoricalFeaturesInfo={}, numIterations=3)
#evaluate(model,trainingData,testData)


# In[ ]:





# In[ ]:
=======
#                                             categoricalFeaturesInfo={}, numIterations=3)
#evaluate(model,trainingData,testData)
>>>>>>> 5d0356ada16b21acb718a3437da55805e6b554ef





# In[ ]:





# In[7]:


#dat=evaluate(model,trainingData,testData)


# In[8]:


#import pandas as pd
#pd.DataFrame(dat).to_csv('resutls.txt',index=False,header=False)


# In[ ]:





# In[ ]:





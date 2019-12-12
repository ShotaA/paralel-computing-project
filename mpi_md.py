from mpi4py import MPI
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


def return_2nd_nonblocking(source,dest,tag,items):
	COMM=MPI.COMM_WORLD
	if COMM.Get_rank()==source:
		req=COMM.isend(items,dest=dest,tag=tag)
		req.wait()
		req=COMM.irecv(source=dest,tag=tag+1)
		data=req.wait()
		return data
	if COMM.Get_rank()==dest:
		req=COMM.irecv(source=source,tag=tag)
		data=req.wait()
		req=COMM.isend(data[1],dest=source,tag=tag+1)
		req.wait()
		return data
	else:
		return None

class RandomForest:
	def __init__(self,n_estims=10,f=50):
	    self.estims=[DecisionTreeClassifier(random_state=0) for x in range(n_estims)]
	    self.f=f
	    self.feats=[]

	def fit(self, X,y):
	    for mod in self.estims:
		feat = np.random.choice(range(X.shape[1]), self.f, replace=False)
		self.feats.append(feat)
		ind = np.random.randint(0,len(y),len(y))
		new_X=X.to_numpy()[ind]
		new_X=new_X[:,feat]
		new_y=y[ind]
		mod.fit(new_X,new_y)
		print(self.feats)

	def predict(self, X):
	    preds=np.zeros((len(self.estims),X.shape[0]))
	    for i in range(len(self.estims)):
		preds[i]=self.estims[i].predict(X.to_numpy()[:,self.feats[i]])
	    preds=preds.sum(axis=0)
	    return (np.where(preds>0,1,-1))
	

   
class BagDT:

	def __init__(self,n_estims=10):
	    self.estims=[DecisionTreeClassifier(random_state=0) for x in range(n_estims)]

	def fit(self, X,y):
	    for mod in self.estims:
		ind = np.random.randint(0,len(y),len(y))
		new_X=X.to_numpy()[ind]
		new_y=y[ind]
		mod.fit(new_X,new_y)

	def predict(self, X):
	    preds=np.zeros((len(self.estims),X.shape[0]))
	    for i in range(len(self.estims)):
		preds[i]=self.estims[i].predict(X)
	    preds=preds.sum(axis=0)
	    return (np.where(preds>0,1,-1))

## LOAD DATA
data=pd.read_csv('blood.csv')
X=data.iloc[:,:-1]
y=data.iloc[:,-1]


model = BagDT(n_estims=50)
model.fit(X,y)

error_rate_train = accuracy_score(model.predict(X),y)
print("Train error:",1-error_rate_train)




#model = RandomForest(n_estims=20,f=1)
#model.fit(X,y)
#error_rate_train = accuracy_score(model.predict(X),y)
#error_rate_test =  accuracy_score(test.y, model.predict(test.drop('y',axis=1)))
#print("Train error:",1-error_rate_train)
#print("Test error:",1-error_rate_test)
	

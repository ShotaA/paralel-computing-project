from mpi4py import MPI
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import math


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
	    self.estim=DecisionTreeClassifier(random_state=0) 
	    self.f=f
	    self.feat=[]

	def fit(self, X,y):
		feat = np.random.choice(range(X.shape[1]), self.f, replace=False)
		#self.feats.append(feat)
		self.feat=feat
		ind = np.random.randint(0,len(y),len(y))
		new_X=X.to_numpy()[ind]
		new_X=new_X[:,feat]
		new_y=y[ind]
		self.estim.fit(new_X,new_y)

	def predict(self, X):
		
		#preds=preds.sum(axis=0)
		comm=MPI.COMM_WORLD
		mpi_rank = comm.Get_rank()
		size = comm.Get_size()
		sendbuf=None


		preds=np.zeros((size,X.shape[0]))
		#for i in range(len(self.estims)):
		#pred=self.estim.predict(X)
		pred=self.estim.predict(X.to_numpy()[:,self.feat])
		comm.Gather(pred,preds,root=0)
		if mpi_rank==0:
			preds=preds.sum(axis=0)
			return (np.where(preds>0,1,-1))
		return None
	

   
class BagDT:
	def __init__(self):
	    self.estim=DecisionTreeClassifier(random_state=0) 

	def fit(self, X,y):
		ind = np.random.randint(0,len(y),len(y))
		new_X=X.to_numpy()[ind]
		new_y=y[ind]
		self.estim.fit(new_X,new_y)

	def predict(self, X):
		comm=MPI.COMM_WORLD
		mpi_rank = comm.Get_rank()
		size = comm.Get_size()
		sendbuf=None


		preds=np.zeros((size,X.shape[0]))
		#for i in range(len(self.estims)):
		pred=self.estim.predict(X)
		comm.Gather(pred,preds,root=0)
		if mpi_rank==0:
			preds=preds.sum(axis=0)
			return (np.where(preds>0,1,-1))
		return None


## LOAD DATA

times=[]
filename='blood.csv'

data=pd.read_csv(filename)
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
f=int(math.log(X.shape[1]+1,2))
#Bagging

comm=MPI.COMM_WORLD
size = comm.Get_size()
mpi_rank = comm.Get_rank()
if mpi_rank==0:
	start=time.time()
model = BagDT()
model.fit(X,y)
if mpi_rank==0:
	taken=time.time()-start
	pd.DataFrame([[filename,taken,size]]).to_csv('bag_mpi.txt',mode='a',index=False,header=False)
	print(taken)

model.predict(X)
#Random Forest

comm=MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
if mpi_rank==0:
	start=time.time()
model = RandomForest(f=f)
model.fit(X,y)
if mpi_rank==0:
	taken=time.time()-start
	print(taken)
	pd.DataFrame([[filename,taken,size]]).to_csv('rf_mpi.txt',mode='a',index=False,header=False)
model.predict(X)




	

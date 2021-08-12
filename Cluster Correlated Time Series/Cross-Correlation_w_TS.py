
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy  as np

def get_corr(Tss:pd.DataFrame) -> np.array:
	
	def calc_corr(X:np.array,Y:np.array) -> float:
		X = (X - np.mean(X)) / (np.std(X) * len(X))
		Y = (Y - np.mean(Y)) / (np.std(Y))

		return np.correlate(X, Y, 'valid')



	Tss  = Tss.iloc[1:,:].values
	Corr = np.zeros(shape=(Tss.shape[1],Tss.shape[1]))

	for i in range(0,Tss.shape[1]):
		for j in range(0,Tss.shape[1]):
			if (i != j):
				Corr[i,j] = calc_corr(Tss[:,i], Tss[:,j])

	return Corr

#Get all Tss that are highly correlated with X 
def get_indices_corr_w(ind:int,thresh:float,Corr_T:np.array) -> np.array:
	return np.where(Corr_T[ind,:]>thresh)

def plot(indices:np.array,Tss:pd.DataFrame) -> None:
	for ind in indices:
		plt.plot(Tss.iloc[1:,ind].values)

	plt.legend(loc='best')
	plt.show()

Tss 	   = pd.read_csv('/home/enihcam/Downloads/github repos/Clustering-In-A-Nutshell/Cluster Correlated Time Series/graphs45.csv')

corr_table = get_corr(Tss)

print(np.amax(corr_table))
indices    = get_indices_corr_w(ind=6,thresh=0.9,Corr_T=corr_table)


plot(indices=indices[0],Tss=Tss)






from scipy import signal
import pandas as pd
import numpy  as np

def get_corr(Tss:pd.DataFrame) -> np.array:
	Tss  = Tss.iloc[1:,:].values
	Corr = np.empty(shape=(Tss.shape[1],Tss.shape[1]))

	for i in range(0,Tss.shape[1]):
		for j in range(0,Tss.shape[1]):
			if (i != j):
				Corr[i,j] = signal.correlate(Tss[:,i], Tss[:,j], mode='valid')

	return Corr

Tss 	   = pd.read_csv('/home/enihcam/Downloads/github repos/Clustering-In-A-Nutshell/Cluster Correlated Time Series/graphs45.csv')
corr_table = get_corr(Tss)


#Get all graphs that are highly correlated with a certain 






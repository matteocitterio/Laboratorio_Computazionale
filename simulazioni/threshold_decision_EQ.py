from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats.stats import pearsonr
import seaborn as sns

#############################################################################################################################################

def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print (ujd, wjd)
        H = 0
 
    return H

#############################################################################################################################################

N=2100
gauss_step=0.01
almost_equally_spaced=np.zeros(N)
equally_spaced=np.linspace(10,100,N)
storico_sigma=np.zeros((100))
F_Nss=[]

fig, ax=plt.subplots()

sogliole=[0.48,0.50,0.52,0.54,0.56,0.58,0.6]
sigma_rels=[0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]

#sogliole=[0.48,0.5]
#sigma_rels=[0.2,0.25,0.3]
for i in sogliole:

	gaussage=0.14

	F_Ns=[]

	for m in sigma_rels:

		print('N=',N)

		pivot=0
		a=0
		b=0
	
		while(pivot==0):

			counter1=0
			counter2=0
			counter3=0


			for j in range(0,100):
			
#				print(j)
				
				gaussati=norm.rvs(loc=90/N, scale=(90/N)*gaussage, size=N)

				for l in range(gaussati.size):

					almost_equally_spaced[l]=equally_spaced[l]+gaussati[l]

				df=pd.DataFrame(almost_equally_spaced)			#repulsione

				h=hopkins(pd.DataFrame(almost_equally_spaced))

				data=np.sort(almost_equally_spaced)
				intervals=np.ediff1d(data)
				storico_sigma[j]=np.std(intervals)/np.mean(intervals)

				if (h<i):

					counter1+=1
	
				elif(h>0.75):

					counter3+=1
		
				else:

					counter2+=1
			
			sigma_rel=np.mean(storico_sigma)			#sigma rel media delle iterazioni
		
			#se la sigma_rel va bene allora andiamo a calcolare F_N, se no cambiamo il gaussaggio
	
			if(sigma_rel>=m and b==1):
	
				F_N=(counter2+counter3)/(counter1+counter2+counter3)			#calcolo e salvo la sigma rel
				F_Ns.append(F_N)
				print('Fatto: F_N',F_N,'soglia h',i,'sigma_rel',m)
				pivot=1
				
			elif(sigma_rel<m and a==1):
	
				F_N=(counter2+counter3)/(counter1+counter2+counter3)			#calcolo e salvo la sigma rel
				F_Ns.append(F_N)
				print('Fatto: F_N',F_N,'soglia h',i,'sigma_rel',m)
				pivot=1
	
			elif(sigma_rel>=m):
		
				print('troppo in alto la sigma, dati:', sigma_rel,'desiderata:',m)
		
				gaussage-=gauss_step
			
				print('New_Gaussage:',gaussage)
				a=1
	
			elif(sigma_rel<m):
		
				print('troppo poco bassa: dati',sigma_rel,'desiderata:',m)
		
				gaussage+=gauss_step
				
				print('New_Gaussage:',gaussage)
				b=1
	
	F_Nss.append(F_Ns)
sns.set(font_scale=2)
sns.heatmap(F_Nss,cmap="YlGnBu",cbar_kws={'label': r'$F_N$ %'},annot=True,linewidths=.5,annot_kws={"fontsize":'medium'})

#print(F_Nss)

plt.xlabel(r'$\sigma_{rel}(N=2100)$',fontsize='medium') 
plt.ylabel("soglia h", fontsize='medium') 

labelsx = sigma_rels
labelsy=sogliole

x = np.arange(0,13)+0.5
y = np.arange(0,7)+0.5

# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labelsx, fontsize ='medium')
plt.yticks(y, labelsy, rotation='horizontal',fontsize='medium')

fig.savefig('/home/matteo/Scrivania/heatmap_vera.pdf')

plt.show()

	









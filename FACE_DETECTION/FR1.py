import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

dfx = pd.read_csv('xdata.csv')
dfy = pd.read_csv('ydata.csv')

x=dfx.values
y=dfy.values

x = dfx.values
y = dfy.values

x = x[:,1:]
y = y[:,1:].reshape((-1,))

#print(x)

print(x.shape)
print(y.shape)

#print(y)

plt.scatter(x[:,0],x[:,1],c=y)
#plt.show()

#Detect the point in two different cluster
query_x = np.array([2,3])
plt.scatter(x[:,0],x[:,1],c=y)
plt.scatter(query_x[0],query_x[1],color='red')
#plt.show()

#function for finding distance from neighborer point
def dist(x1,x2):
	return np.sqrt(sum((x1-x2)**2))

#function for defining the pridiction of point
def knn(x,y,querypoint,k=5):
    
    vals = []
    m = x.shape[0]

    for i in range(m):
        d = dist(querypoint,x[i])
        vals.append((d,y[i]))

    vals = sorted(vals)
    #Nearest/First K points 
    vals = vals[:k]
    vals = np.array(vals)

    #print(vals)

    new_vals = np.unique(vals[:,1],return_counts=True)
    print(new_vals)

    index = new_vals[1].argmax()
    pred = new_vals[0][index]

    return pred


#print(knn(x,y,query_x))

x = knn(x,y,[0,0])
print(x)       	
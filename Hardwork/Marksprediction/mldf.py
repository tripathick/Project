#MAchine LEarning 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# load and visualize the data:dowload>load>visualize>normalisation


x=pd.read_csv('Linear_X_Train.csv')
y=pd.read_csv('Linear_Y_Train.csv')

#print(x.head())

#print(y.head())

# convert x,y to numpy arrays:  

x=x.values
y=y.values

#Normalisation;

u = x.mean()
std=x.std()

#print(u,std)



# Visualize..


plt.style.use('seaborn')
plt.scatter(x,y,color='Green')

plt.title(" HARDWORKS vs PERFORMANCE GRAPH..")
plt.xlabel("Hardwork")
plt.ylabel("Performance")

#plt.show()

#print how many point on x-axis,y-axis
#x.shape()
#y.shape()

#................................
#section : 2 Linear Regresssion Algorithm

def hypothesis(x,theta):
	y_ = theta[0] + theta[1]*x   #y_= value of y hat
	return y_

def gradient(X,Y,theta):  #X,Y denotes all entire dataset	
    
    m = X.shape[0]

    grad = np.zeros((2,))

    for i in range(m):
    	x= X[i]

    	y_= hypothesis(x,theta)
    	y = Y[i]

    	grad[0] += (y_ - y)
    	grad[1] += (y_ - y)*x

    return grad/m


def error(X,Y,theta):
    
    m=X.shape[0]
    total_error=0.0

    for i in range(m):
        y_ = hypothesis(X[i],theta)
        total_error += (y_ - Y[i]) ** 2

    return total_error/m        	

def gradientDescent(X,Y,max_steps=100,learning_rate =0.1):
    
    theta = np.zeros((2,))
    error_list = []

    for i in range(max_steps):
        #compute grad

        grad = gradient(x,y,theta) 
        
        e = error(X,Y,theta)
        error_list.append(e)
        #Update theta

        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]

    return theta,error_list


theta,error_list = gradientDescent(x,y) 

#
#print(error_list)

#plt.plot(error_list)  

#plt.title("Reduction error over time")       
#plt.show()

#.................................
#SECTION :3: Prediction and Best Line

y_ = hypothesis(x,theta)
#print(y_)


# Training + Prediction 

#plt.scatter(x,y)

#plt.plot(x, y_ , color='red' , label="prdiction")
plt.legend()
#plt.show()


#LOAD THE TEST DATA

X_test = pd.read_csv('Linear_X_Test.csv').values
y_test = hypothesis(X_test,theta)

#print(y_test.shape)

#create a data frame

df = pd.DataFrame(data= y_test,columns=["y"])

df.to_csv('y_prediction.csv' , index=False)

#...............................

#section : 4: Computing score

# Score : R2 (R-squared) Or Cofficient of Determination

def r2_score(y,y_):
    
    # Instead of loop ,np.sum is recomended as it is fast
	num = np.sum((y-y_)**2)
	denom = np.sum((y-y.mean())**2)

	score = (1- num/denom)

	return score*100

print(r2_score(y,y_))



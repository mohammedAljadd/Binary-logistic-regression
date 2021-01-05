# Binary logistic regression

  Hello everyone, in this project we will implement binary logistic regression to predict whether someone is gonna be admitted or not based on two given exams score. This project will be divided into two parts. In the first part, we will code everything, so we will not use the Scikit learn library and we will write all the necessary functions like cost function or gradient descent. In the second part, we will use from Scikit learn library and you will see that the problem can be solved in a few lines of code.
  
  The implementation was done using Python. You will find the jupyter notebook version in this repository.


# 1st Part:
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
First, let's present our dataset. Our dataset contains three columns, the first two columns correspond to the score of both exams with a range of about 30 to 100. The thired column is the output vector that contains values of 0s and 1s. For example, if y (i) = 1 then the ith entry belongs to class 1, class of admitted students.

# Libraries

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
# Dataset loading and variables initialization :

    data = pd.read_csv('data1.txt',header=None)
    data = data.to_numpy()
    m = data.shape[0]
    x1 = data[:,0].reshape((m, 1))/100
    x2 = data[:,1].reshape((m, 1))/100
    x = np.hstack((np.ones(x1.shape),x1,x2 ))
    y = data[:,2].reshape((m, 1))
    theta = np.random.rand(3,1).reshape(3,1)
    alpha = 0.15
    itterations = 2000
    error_history = np.zeros((itterations)).reshape(itterations,1)

Let's split data according to the two classes :
    
    x1_admitted = x1[x1!=x1-y]
    x2_admitted = x2[x2!=x2-y]
    x1_not_admitted = x1[x1==x1-y]
    x2_not_admitted = x2[x2==x2-y]

# All the necessary functions :

    def sigmoid(z):
        sig = 1 / (1 + np.exp(-z))   
        sig = np.minimum(sig, 0.999999999999)  
        sig = np.maximum(sig, 0.000000000001)  
        return sig

    def error(theta):
        return -1/m * np.sum(np.multiply(y,np.log(sigmoid(x.dot(theta)))) + (1-y)*np.log(1-sigmoid(x.dot(theta))))

    def gradient(theta):
        return 1/m * x.T.dot((sigmoid(x.dot(theta))-y))

    def gradientDescent(alpha,itterations,theta):
        for i in range(itterations):
            error_history[i] = error(theta)
            theta = theta - alpha*gradient(theta)
        return theta,error_history


    def predict(x,y,theta):
          probability = sigmoid(x.dot(theta))
        return np.round(probability)









# Scikit-learn implementation:

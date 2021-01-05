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

# Plotting data :


    plt.scatter(x1_add,x2_add,label='Admitted',marker="+",color='g')
    plt.xlabel('first exam')
    plt.ylabel('second exam')
    plt.title('training data')
    plt.scatter(x1_not_add,x2_not_add,label='Not Admitted',marker="x",color='r')
    plt.legend()

![alt text](https://github.com/mohammedAljadd/Two-class-logistic-regression/blob/main/plots/data.PNG)

# Plot of the cost function according to number of itterations


    plt.plot(error_history)
    plt.xlabel('number of itterations')
    plt.ylabel('Cost function J')
    plt.show()

![alt text](https://github.com/mohammedAljadd/Two-class-logistic-regression/blob/main/plots/jhist.PNG)

# Plot the decision boundary
The decision boundary verify θ0 + θ1.x1 + θ2.x2 = 0

    plot_x = np.array([np.min(x[:,1]),  np.max(x[:,1])])
    plot_y = 1/optimum[2] * (-optimum[0]-optimum[1]*plot_x)
    plt.plot(plot_x,plot_y,label='Decision boundary')
    plt.scatter(x1[x1!=x1-y],x2[x2!=x2-y],label='Admitted',marker="+",color='g')
    plt.xlabel('first exam')
    plt.ylabel('second exam')
    plt.title('training data')
    plt.scatter(x1[x1==x1-y],x2[x2==x2-y],label='Not Admitted',marker='x',color='r')
    plt.legend()
    
![alt text](https://github.com/mohammedAljadd/Two-class-logistic-regression/blob/main/plots/boundary.PNG)    

# Model evaluation :
1) Model accuracy :

        predictions = predict(x,y,optimum)
        print('Accuracy of your model is ',np.sum(predictions==y)/len(y)*100,'%')

Accuracy of your model is  93.0 %.

2) Confusion matrix :

![alt text](https://miro.medium.com/max/576/1*BAAk374bKlraxnJvV3_hyg.png)

      all = np.hstack((x1,x2,y,predictions))
      c = np.zeros((2,2))
      c[0,0] = sum( (all[:,2]==all[:,3])[all[:,3]==0]) 
      c[1,1] = sum( (all[:,2]==all[:,3])[all[:,3]==1])
      c[0,1] = sum( (all[:,2]!=all[:,3])[all[:,2]==0])
      c[1,0] = sum( (all[:,2]!=all[:,3])[all[:,2]==1])
      print(c)

[34 ,6] <br/>
[1 ,59]


# 2nd Part : Scikit-learn implementation:

Let's import logistic regression model from sklearn :'

      from sklearn.linear_model import LogisticRegression
      logisticRegr = LogisticRegression()
      x_train = np.hstack((x1,x2))
      y = y.ravel()
      logisticRegr.fit(x_train, y)
      intercept = logisticRegr.intercept_
      coefs = logisticRegr.coef_
      sklearn_optimum = np.vstack((intercept,coefs.reshape(2,1)))
      
The confusion matrix :

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, predictions)
    print(cm)
    
[34 ,6] <br/>
[1 ,59]

The roc curve :

A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. 


![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Roc-draft-xkcd-style.svg/800px-Roc-draft-xkcd-style.svg.png)
    
    from sklearn.metrics import roc_curve
    X = x
    y_pred_proba= sigmoid(X.dot(optimum))
    fpr,tpr,thresholds=roc_curve(y,y_pred_proba)
    plt.plot([0, 1],[0, 1],'--')
    plt.plot(fpr,tpr,label='linear regression')
    plt.xlabel('false positive')
    plt.ylabel('false negative')
    plt.show()
    
![alt text](https://github.com/mohammedAljadd/Two-class-logistic-regression/blob/main/plots/roc.PNG)


The decision boundary :

    plot_x = np.array([np.min(x[:,1]),  np.max(x[:,1])])*100
    plot_y = 1/sklearn_optimum[2] * (-sklearn_optimum[0]-sklearn_optimum[1]*plot_x)
    plt.plot(plot_x,plot_y,label='Decision boundary')
    plt.scatter(x1_add*100,x2_add*100,label='Admitted',marker="+",color='g')
    plt.xlabel('first exam')
    plt.ylabel('second exam')
    plt.title('training data')
    plt.scatter(x1_not_add*100,x2_not_add*100,label='Not Admitted',marker="x",color='r')
    plt.legend()
    
![alt text](https://github.com/mohammedAljadd/Two-class-logistic-regression/blob/main/plots/boundarySK.PNG)

The model accuracy :

    X_test = np.hstack((x1,x2))
    R_test=logisticRegr.score(X_test,y)*100
    print('Accuracy of your model is ',R_test,'%')
Accuracy of your model is  91.0 %


If you have had a problem or encountered a bug, do not hesitate to contact me: aljadd.mohammed@inemail.ine.inpt.ma

I will be happy to receive a feedback :blush: .

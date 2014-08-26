'''
Please see LICENSE for copyright
Author: 'Wu Li' <li.koun@gmail.com>

The implementation of logistic regression. 

'''
import numpy as np

def sigmoid (z):
    return (1 / (1 + np.exp(-z)))

def costLogistic(X, y, theta, lmd):
    '''
    Cost function. The return value is a tuple contains cost and gradient descent.
    '''
    t = theta[1:]
    m = X.shape[0]
    g = sigmoid(np.dot(X, theta))
    J = (-np.dot(y, np.log(g)) - np.dot(1-y, np.log(1-g)))/m + lmd*np.dot(t,t.T)/m
    GRAD = np.dot(X.T, g-y)/m 
    return (J, GRAD)


def gradientDescent(X, y, theta, lmd, iterations):
    '''
    Gradient descent 
    '''
    J_hist = np.zeros(iterations)
    for i in range(iterations):
        J_hist[i], GRAD = costLogistic(X, y, theta, lmd)
        theta -= GRAD

    return (J_hist, theta)


def oneVsAll(X, y, lmd, numLabels):
    '''
    Calculates the weights for each number
    '''
    n = X.shape[1]
    allTheta = np.zeros((n, numLabels))
    Y = np.eye(10)[y.astype(int),:]
    for i in range(numLabels):
        theta = np.ones(n)
        J_hist, theta = gradientDescent(X, Y[:,i], theta, lmd, 100)
        allTheta[:,i] = theta

    return allTheta


def predict (X, allTheta):
    '''
    Predict
    '''
    V = sigmoid(np.dot(X, allTheta))
    idx = np.argmax(V, 1)
    return idx

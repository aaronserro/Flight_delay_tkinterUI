import numpy
import matplotlib

import math

def sigmoid(z):
    '''
    takes math real number input and displays it in a range between 0 and 1 - probability


    '''
    z = max(min(z, 500), -500)
    output = 1/(1+math.exp(-z))
    return output


def predict(X, weights, bias):
    """
    This function will take in X - which is the matrix of data,
    weights - the assigned weight of each feature in the dataset
    bias - the general offset of the feature

    This functions outputs the probability that the probabilities for each row in X


    """
    z = 0
    sig = 0
    arr = []
    for i in range(0, len(X)):
        for j in range(0, len(X[i])):
            z += weights[j]*X[i][j]
        z += bias
        sig = sigmoid(z)
        arr.append(sig)
        z = 0
    return arr


def compute_loss(y_true, y_pred):
    """
    How far off the predictions are from the actual answers
    ex: If the model predicts that a flight will be delayed and it is not,
    this will assign a penalty based on how wrong the model was
    y_true is the actual value and y_pred is the real value
    it outputs a single number which represents the inaccuracy


    Binary Cross entropy(Log loss)
    """
    epsilon = 1e-15
    average_loss = 0
    for i in range(len(y_true)):
        y_hat = min(max(y_pred[i], epsilon), 1 - epsilon)
        loss = -(y_true[i]*math.log(y_hat)+((1-y_true[i]) * math.log(1-y_hat)))
        average_loss += loss

    average_loss = average_loss/len(y_true)
    return average_loss


def compute_gradients(X, y_true, y_pred):
    '''
    How does the model adjust its weights to do better next time
    This is where the model learns how to crrect itself.

    '''

    sum_error = 0
    dw = [0]*len(X[0])
    for i in range(0, len(y_true)):
        err_row = y_pred[i]-y_true[i]
        for j in range(0, len(X[0])):
            weight_update = X[i][j]*err_row

            dw[j] += weight_update
        sum_error += err_row
    db = sum_error/len(X)

    return dw, db


def update_parameters(weights, bias, dw, db, lr):
    '''
    takes the gradients and updates them based on their values
    subtract the gradient multiplied by the learning rate
    dw, db: gradient directions
    lr: how big each learning step is


    '''
    for i in range(0, len(weights)):
        weights[i] = weights[i]-lr*dw[i]
    bias = bias - lr*db
    return weights, bias

import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sys.path.append("..")  # This adds the parent folder (root) to the import path
from src.model import predict, compute_gradients, compute_loss, update_parameters


#Loading the Information
weights = np.random.uniform(-1, 1, size=7)
bias = np.random.uniform(-1, 1)
Whole = pd.read_csv('../data/Flight_Delay_Dataset_with_weather.csv')
labels = Whole['delayed']
labels = labels.to_numpy()
to_train = Whole
to_train.drop(columns=['delayed'], axis=1, inplace=True)
to_train = to_train.to_numpy()
to_train = scaler.fit_transform(to_train)

#Training the Model
num_epochs = 1000
for num in range(num_epochs):
    predicitons = predict(to_train, weights, bias)
    loss = compute_loss(labels, predicitons)
    dw, db = compute_gradients(to_train, labels, predicitons)
    lr = 0.1
    weights, bias = update_parameters(weights, bias, dw, db, lr)
    if num % 100 == 0:
        print(f"Epoch {num} | Loss: {loss}")
        #print(f"dw: {dw}")
        #print(f"db: {db}")
        #print(weights)
        #print(bias)

np.save("../models/weights.npy", weights)
np.save("../models/bias.npy", bias)







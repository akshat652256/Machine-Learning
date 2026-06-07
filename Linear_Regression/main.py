import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Data
dataset = pd.read_csv("/kaggle/input/datasets/akshatsatyanarayan/cartest/carprice.csv")

# Data Preprocessing

X = dataset.drop(columns=["car_ID","CarName","price"])
X = pd.get_dummies(X,columns=["fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"],dtype = int)
X = X.to_numpy()

y = dataset["price"].to_numpy()

# Train - Test Split

np.random.seed(42)

split = int(0.8 * len(X))
indices = np.random.permutation(len(X))

train_indices = indices[:split]
test_indices = indices[split:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

# Standardize

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std==0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Initialize Parameters and Hyperparameters

m = X_train.shape[0]
n = X_train.shape[1]

w = np.random.randn(n)
b = 0.0

alpha = 0.01
epochs = 500

# Training Loop

loss_history = []
for epoch in range(epochs):
    error = 0.0
    dw = np.zeros_like(w)
    db = 0.0
    for row in range(m):
        y_pred = np.dot(w, X_train[row]) + b
        pred_error = (y_pred - y_train[row])
        error += pred_error ** 2
        dw += pred_error * X_train[row]
        db += pred_error
    loss = error / (2 * m)
    dw = dw / m
    db = db / m
    w = w - alpha * dw
    b = b - alpha * db

    loss_history.append(loss)
    
    print(f"Epoch: {epoch+1}, Loss: {loss}")

# Testing the Model

ss_res = 0.0
ss_tot = 0.0
test_loss = 0.0
for row in range(m):
    y_test_pred = np.dot(w,X_test) + b
    y_test_error = y_test_pred - y_test[row]
    y_test_error_mean = y_test_pred - y_test.mean()
    ss_res += y_test_error ** 2
    ss_tot += y_test_error_mean ** 2
    test_loss += y_test_error ** 2

test_loss = test_loss / 2 * m
r2_score = 1 - (ss_res/ss_tot)

print(f"Test loss: {test_loss}, R2 score: {r2_score}")

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epochs vs Training Loss")
plt.show()
    















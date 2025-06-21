import numpy as np
import matplotlib.pyplot as plt

# 1. Sample dataset (created two seperate arrays for both parts of graph, X n Y) 
X = np.array([100, 200, 300, 400, 500])         # size
Y = np.array([1000.5, 2000.5, 3000.5, 4000.0, 5000.0])  # price

m = len(X)  # the M variable for MSE calculations, dataset (n in math)

# linear Regression Model = refering to general formulae of striaght + MSE later on - (add gradient descent ki things later on)
def predict(X, w, b):
    return w * X + b

# cost function computation time 
def compute_cost(X, Y, w, b):
    total_cost = 0
    for i in range(m):
        y_hat = predict(X[i], w, b)
        total_cost += (y_hat - Y[i])**2
    return total_cost / (2 * m)

# example value to run first case to TRAIN the model
w = 0.9  #  slope, weight (effect)
b = 0.6  # intercept , bias

# predictions using this (w, b)
Y_pred = predict(X, w, b)

# valculate cost
cost = compute_cost(X, Y, w, b)
print(f"Predictions: {Y_pred}")
print(f"Actual Values: {Y}")
print(f"Cost (MSE): {cost}")

# visualization (Plot)
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='red', label='Training Data')
plt.plot(X, Y_pred, color='blue', label=f'Prediction Line (w={w}, b={b})')
plt.xlabel('Size of House (1000 sq.ft)')
plt.ylabel('Price (lakh rupees)')
plt.title('Linear Regression Prediction vs Actual Data')
plt.legend()
plt.grid(True)
plt.show()

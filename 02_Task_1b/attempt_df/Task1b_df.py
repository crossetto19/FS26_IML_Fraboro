# %% [markdown]
# #### General guidance
# 
# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# This is the jupyter notebook version of the template. For the python file version, please refer to the file `template_solution.py`.
# 
# First, we import necessary libraries:

# %%
import numpy as np
import pandas as pd

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

# %% [markdown]
#  #### Loading data

# %%
data = pd.read_csv("../train.csv")
y = data["y"].to_numpy()
data = data.drop(columns=["Id", "y"])
# print a few data samples
print(data.head())
X = data.to_numpy()

# %% [markdown]
# #### Transform features

# %%
"""
Transform the 5 input features of matrix X (x_i denoting the i-th component of a given row in X) 
into 21 new features phi(X) in the following manner:
5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
1 constant feature: phi_21(X)=1

Parameters
----------
X: matrix of floats, dim = (700,5), inputs with 5 features

Compute
----------
X_transformed: matrix of floats: dim = (700,21), transformed input with 21 features
"""
X_transformed = np.zeros((700, 21))
# TODO: Enter your code here

# First with loop
for i in range(X_transformed.shape[0]):
    # Linear feautes x1-x5
    X_transformed[i,0:5] = X[i,0:5]

    # Quadrateic features x6-x10
    X_transformed[i,5:10] = X[i,0:5]**2

    # Exponential features x11-x15
    X_transformed[i,10:15] = np.exp(X[i,0:5])

    # Cosine features x16-x20
    X_transformed[i,15:20] = np.cos(X[i,0:5])

    # Constant x21
    X_transformed[i,20] = 1

assert X_transformed.shape == (700, 21)

# %% [markdown]
# #### Logistic Loss

# %%
# Function to calculate the logistic loss
def logLoss(y, X, w):

    # Calculate the probability, sigmoid
    yhat = 1 / (1 + np.exp(-(X @ w)))

    # Handle 0 and 1 of y, adding a small number to handle cases of log(0)
    small = 1e-15
    loss = -np.mean(y * np.log(yhat + small) + (1 - y) * np.log(1 - yhat + small))

    return loss

# Gradient of the logistic loss for gradient descent
def logLossGrad(y, X, w):
    
    # Calculate the probability, sigmoid
    yhat = 1 / (1 + np.exp(-(X @ w)))

    grad = (X.T @ (yhat - y)) / X.shape[0]

    return grad

# %% [markdown]
# #### Fit data

# %%
"""
Use the transformed data points X_transformed and fit the logistic regression on this 
transformed data. Finally, compute the weights of the fitted logistic regression. 

Parameters
----------
X_transformed: array of floats: dim = (700,21), transformed input with 21 features
y: array of integers \in {0,1}, dim = (700,), input labels

Compute
----------
w: array of floats: dim = (21,), optimal parameters of logistic regression
"""
weights = np.zeros((21,))
# TODO: Enter your code here

# Track the loss values
lossVal = []

# Parameter
stepsize = 1
beta = 0.97
eta = 0.0005
epoch = 0
# With momentum
momentum = np.zeros((21,))

while True:
    preWeights = weights.copy()

    # Current gradient
    grad = logLossGrad(y, X_transformed, weights)

    # Add momentum
    momentum = (beta * momentum) + (stepsize * grad)
    weights = weights - momentum

    currentLoss = logLoss(y, X_transformed, weights)
    lossVal.append(currentLoss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {currentLoss:.4f}")

    if (np.linalg.norm(weights - preWeights) < eta):
        break

    epoch += 1

assert weights.shape == (21,)

# %%
# Check the predictions
final_probabilities = 1 / (1 + np.exp(-(X_transformed @ weights)))
predictions = (final_probabilities >= 0.5).astype(int)
accuracy = np.mean(predictions == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# %%
# Save results in the required format
np.savetxt("./results_df.csv", weights, fmt="%.12f")

# %%




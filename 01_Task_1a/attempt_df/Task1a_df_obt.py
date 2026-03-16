# %% [markdown]
# #### General guidance
# 
# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# This is the jupyter notebook version of the template. For the python file version, please refer to the file `template_solution.py`.
# 
# First, we import necessary libraries:

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

# %% [markdown]
#  #### Loading data

# %%
data = pd.read_csv("../train.csv")
y = data["y"].to_numpy()
data = data.drop(columns="y")
# print a few data samples
print(data.head())

# %% [markdown]
# #### Calculating the average RMSE

# %%
def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    rmse: float: dim = 1, RMSE value
    """
    rmse = 0
    # TODO: Enter your code here

    # Manual approach
    # for i in range(len(y)):
    #     rmse += (y[i] - w@X[i])**2

    # rmse = (rmse/len(y))**(1/2)

    # Faster in one line
    rmse = np.sqrt(np.mean((y - X @ w)**2))

    assert np.isscalar(rmse)
    return rmse

# %% [markdown]
# #### Fitting the regressor

# %%
def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    weights = np.zeros((13,))
    # TODO: Enter your code here

    # Closed form
    # weights = np.linalg.inv(X.T@X + lam*np.identity(X.shape[1]))@X.T@y

    # More stable version
    weights = np.linalg.solve(
        X.T @ X + lam * np.identity(X.shape[1]),
        X.T @ y
    )

    assert weights.shape == (13,)
    return weights

# %% [markdown]
# #### Performing computation

# %%
"""
Main cross-validation loop, implementing 10-fold CV. In every iteration 
(for every train-test split), the RMSE for every lambda is calculated, 
and then averaged over iterations.

Parameters
---------- 
X: matrix of floats, dim = (150, 13), inputs with 13 features
y: array of floats, dim = (150, ), input labels
lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV

Compute
----------
avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
"""
X = data.to_numpy()
# The function calculating the average RMSE
lambdas = [0.1, 1, 10, 100, 200]
n_folds = 10

RMSE_mat = np.zeros((n_folds, len(lambdas)))

# TODO: Enter your code here. Hint: Use functions 'fit' and 'calculate_RMSE' with training and test data
# and fill all entries in the matrix 'RMSE_mat'

"""
Notes:
1. Iterate over all lambdas
2. Each time cross validate 10 times
3. Calculate the mean RMSE for each lambda
"""
# Move outside of loop and set shuffle with seed
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Iterate over all lambdas
for i, current_lam in enumerate(lambdas):
    
    # Each time cross validate 10 times
    for j, (train, test) in enumerate(kf.split(X)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        # Calculate the weights based on the splits
        weights = fit(X_train, y_train, current_lam)
        current_RMSE = calculate_RMSE(weights, X_test, y_test)

        RMSE_mat[j, i] = current_RMSE

avg_RMSE = np.mean(RMSE_mat, axis=0) # avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
assert avg_RMSE.shape == (5,)

# %%
# Save results in the required format
np.savetxt("./results_df_opt.csv", avg_RMSE, fmt="%.12f")



# %% [markdown]
# # Task 2
# This serves as a template which will guide you through the implementation of this task. It is advised to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# This is the jupyter notebook version of the template. For the python file version, please refer to the file `template_solution.py`.

# %% [markdown]
# First, we import necessary libraries:

# %%
import numpy as np
import pandas as pd

# Add any other imports you need here
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

# %% [markdown]
# # Data Loading
# TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
# (and potentially change initialization of variables to accomodate how you deal with non-numeric data)

# %%
"""
This loads the training and test data, preprocesses it, removes the NaN
values and interpolates the missing data using imputation

Parameters
----------
Compute
----------
X_train: matrix of floats, training input with features
y_train: array of floats, training output with labels
X_test: matrix of floats: dim = (100, ?), test input with features
"""
# Load training data
train_df = pd.read_csv("../train.csv")
    
print("Training data:")
print("Shape:", train_df.shape)
print(train_df.head(2))
print('\n')
    
# Load test data
test_df = pd.read_csv("../test.csv")

print("Test data:")
print(test_df.shape)
print(test_df.head(2))

# Dummy initialization of the X_train, X_test and y_train   
# TODO: Depending on how you deal with the non-numeric data, you may want to 
# modify/ignore the initialization of these variables

# Drop rows where no value for CHF is given
train_df = train_df[train_df["price_CHF"].notna()]

# Split into X and y train
y_train = train_df['price_CHF']
X_train_raw = train_df.drop(['price_CHF'], axis=1)
X_test_raw = test_df.copy()

# Convert the seasons into numerical values
X_train_num = pd.get_dummies(X_train_raw, columns=["season"], dtype=int)
X_test_num = pd.get_dummies(X_test_raw, columns=["season"], dtype=int)

# Applying iterative imputer to handle nan in train data
imp = IterativeImputer(max_iter=10, random_state=0)
X_train_imp = imp.fit_transform(X_train_num) 
X_test_imp = imp.transform(X_test_num)

# TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
X_train = X_train_imp
y_train = y_train.to_numpy()
X_test = X_test_imp

assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"

# %%
"""
Notes for data processing:
- Use iterative imputer as mentioned in the description on train and test data
    - Maybe later to refine the training, apply iterative imputer only on rows from the same season?
- But drop any row where no value for CHF is given, since nothing can be learned from that
"""

# # Convert the season into numerical values
# test_train_df = train_df.copy()
# test_train_df_dumm = pd.get_dummies(test_train_df, columns=["season"], dtype=int)

# # Drop rows where no value for CHF is given
# test_train_df_dumm = test_train_df_dumm[test_train_df_dumm["price_CHF"].notna()]

# # Applying iterative imputer to handle nan in train data
# imp = IterativeImputer(max_iter=10, random_state=0)
# test_train_df_imp = imp.fit_transform(test_train_df_dumm)

# # Convert back into a df
# test_train_df_clean = pd.DataFrame(
#     data=test_train_df_imp, 
#     columns=test_train_df_dumm.columns, 
#     index=test_train_df_dumm.index
# )

# test_train_df_clean


# %% [markdown]
# # Modeling and Prediction
# TODO: Define the model and fit it using training data. Then, use test data to make predictions

# %%
"""
This defines the model, fits training data and then does the prediction
with the test data 

Parameters
----------
X_train: matrix of floats, training input with 10 features
y_train: array of floats, training output
X_test: matrix of floats: dim = (100, ?), test input with 10 features

Compute
----------
y_test: array of floats: dim = (100,), predictions on test set
"""
class Model(object):
    def __init__(self, kernel=DotProduct()):
        super().__init__()
        self._x_train = None
        self._y_train = None

        # Allow to choose which kernel is used and add variance with alpha
        self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.2)

        # Initate a scaler to better handle the data
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train
        # Use the scaler to transfrom and fit the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.gpr.fit(X_train_scaled, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        #TODO: Use the model to make predictions y_pred using test data X_test
        # Use the scaler to transform the data
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.gpr.predict(X_test_scaled)

        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# %%
# Create the scaled version of X_train for testing
cv_scaler = StandardScaler()
X_train_cv_scaled = cv_scaler.fit_transform(X_train)

# Different alpha values to test
for j in [0.1, 0.2, 0.3, 0.5]:
    print("Current alpha value:", j)
    
    # Test how well the different kernel fit the train data
    for i in [DotProduct(), RBF(), Matern(), RationalQuadratic()]:
        
        # Add variance with alpha
        test_gpr = GaussianProcessRegressor(kernel=i, alpha=j)
        
        # Run cross-validation
        scores = cross_val_score(test_gpr, X_train_cv_scaled, y_train, cv=5, scoring='r2')
        
        print(i)
        print(scores.mean())


# %%
model = Model(RationalQuadratic())
# Use this function to fit the model
model.fit(X_train=X_train, y_train=y_train)
# Use this function for inference
y_pred = model.predict(X_test)

# %% [markdown]
# # Saving Results
# You don't have to change this

# %%
dt = pd.DataFrame(y_pred) 
dt.columns = ['price_CHF']
dt.to_csv('results_df.csv', index=False)
print("\nResults file successfully generated!")

# %%




import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline

def load_data():
    # Load raw data
    train_df = pd.read_csv("train.csv").dropna(subset=["price_CHF"])
    test_df = pd.read_csv("test.csv")

    # Extract target and features
    y_train = train_df['price_CHF'].to_numpy()
    X_train_raw = train_df.drop(['price_CHF'], axis=1)
    X_test_raw = test_df.copy()

    # One-hot encode seasons
    X_train = pd.get_dummies(X_train_raw, columns=["season"], dtype=int)
    X_test = pd.get_dummies(X_test_raw, columns=["season"], dtype=int)

    # Sanity check for dimensions
    assert X_train.shape[1] == X_test.shape[1]
    
    return X_train, y_train, X_test

class Model:
    def __init__(self):
        self.best_pipeline = None

    def _find_best_params(self, X_train, y_train):
        
        # Calculate features for ARD (including missing value indicators)
        n_features = X_train.shape[1] + int(X_train.isnull().any().sum())
        ard_scales = np.ones(n_features)

        kernels = [
            DotProduct()**8,
            DotProduct()**16,
            RBF(length_scale=ard_scales, length_scale_bounds=(1e-5, 1e8)), 
            Matern(length_scale=ard_scales, length_scale_bounds=(1e-5, 1e8)), 
            RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-5, 1e8)),
            Matern(length_scale=ard_scales, length_scale_bounds=(1e-5, 1e8)) + RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-5, 1e8))
        ]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_score = -np.inf
        best_kernel, best_alpha = None, 0.01

        print("Searching for best hyperparameters...")
        for alpha in [0.01, 0.05, 0.1]:
            for kernel in kernels:
                # Build temporary pipeline for testing
                temp_pipe = make_pipeline(
                    KNNImputer(n_neighbors=10, weights="distance", add_indicator=True),
                    StandardScaler(),
                    GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=1)
                )
                
                # Evaluate
                scores = cross_val_score(temp_pipe, X_train, y_train, cv=kf, scoring='r2')
                print(f"Tested Alpha={alpha} | Kernel: {kernel} | R2 Score: {scores.mean():.4f}")

                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_kernel, best_alpha = kernel, alpha

        print(f"Winner: R2={best_score:.4f} with {best_kernel} and alpha={best_alpha}")
        return best_kernel, best_alpha

    def fit(self, X_train, y_train):
        best_k, best_a = self._find_best_params(X_train, y_train)
        
        # Build and fit the final pipeline
        self.best_pipeline = make_pipeline(
            KNNImputer(n_neighbors=10, weights="distance", add_indicator=True),
            StandardScaler(),
            GaussianProcessRegressor(kernel=best_k, alpha=best_a, normalize_y=True, n_restarts_optimizer=1)
        )
        self.best_pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        if self.best_pipeline is None:
            raise ValueError("Model must be fitted before predicting.")
        return self.best_pipeline.predict(X_test)

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.utils import resample  # Added import for resample
import joblib

class BootstrapCalibratedClassifier:
    def __init__(self, n_bootstrap_samples=8, base_model_params=None, gpu_id=0):
        """
        Initializes the BootstrapCalibratedClassifier.

        Parameters:
        - n_bootstrap_samples (int): Number of bootstrap samples/models.
        - base_model_params (dict): Parameters for the XGBoost base model.
        - gpu_id (int): GPU device ID to use.
        """
        self.n_bootstrap_samples = n_bootstrap_samples
        self.bootstrap_models = []
        self.calibrated_models = []
        self.gpu_id = gpu_id  # GPU device ID

        # Default XGBoost parameters with GPU support
        self.base_model_params = base_model_params if base_model_params else {
            'tree_method': "hist",
            'enable_categorical': True,
            'max_depth': 10,
            'learning_rate': 0.09937420876401226,
            'n_estimators': 12,
            'gamma': 0.340466985869831,
            'subsample': 0.7222619026651159,
            'colsample_bytree': 0.5739321839530654,
            'reg_alpha': 0.9462720081810914,
            'reg_lambda': 0.5567871265347748
            }

    def fit(self, X_train, y_train):
        """
        Fits the bootstrap and calibrated models on the training data.

        Parameters:
        - X_train (array-like): Training features.
        - y_train (array-like): Training labels.
        """
        for i in range(self.n_bootstrap_samples):
            print(f"Training bootstrap model {i+1}/{self.n_bootstrap_samples} on GPU {self.gpu_id}...")
            
            # Bootstrap sample generation
            X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=len(X_train), replace=True)
            
            # Train base model
            model = xgb.XGBClassifier(**self.base_model_params)
            model.fit(X_train_sample, y_train_sample)
            self.bootstrap_models.append(model)
            
            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_train_sample, y_train_sample)
            self.calibrated_models.append(calibrated_model)
            print(f"Bootstrap model {i+1} trained and calibrated.")

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X (array-like): Input features.

        Returns:
        - array-like: Predicted class labels.
        """
        if not self.bootstrap_models:
            raise Exception("The model has not been fitted yet.")
        
        # Aggregate predictions from bootstrap models
        preds = np.array([model.predict(X) for model in self.bootstrap_models])
        return np.round(np.mean(preds, axis=0)).astype(int)

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Parameters:
        - X (array-like): Input features.

        Returns:
        - array-like: Predicted class probabilities.
        """
        if not self.calibrated_models:
            raise Exception("The model has not been fitted yet.")
        
        # Aggregate calibrated predicted probabilities from bootstrap models
        proba = np.array([calibrated_model.predict_proba(X) for calibrated_model in self.calibrated_models])
        # Average probabilities across all bootstrap models
        return np.mean(proba, axis=0)

    def score(self, X_test, y_test):
        """
        Calculates the accuracy score of the model.

        Parameters:
        - X_test (array-like): Test features.
        - y_test (array-like): True labels.

        Returns:
        - float: Accuracy score.
        """
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

    def save_model(self, filename):
        """
        Saves the trained models to files.

        Parameters:
        - filename (str): Base filename to save the models.
        """
        for idx, model in enumerate(self.bootstrap_models):
            model_path = f"{filename}_model_{idx}.pkl"
            joblib.dump(model, model_path)
        
        # Save calibrated models
        for idx, calibrated_model in enumerate(self.calibrated_models):
            calibrated_model_path = f"{filename}_calibrated_{idx}.pkl"
            joblib.dump(calibrated_model, calibrated_model_path)
        print(f"Models saved to {filename}_model_*.pkl and {filename}_calibrated_*.pkl")

    def load_model(self, filename):
        """
        Loads both base and calibrated models from files.

        Parameters:
        - filename (str): Base filename from which to load the models.
        """
        self.bootstrap_models = []
        self.calibrated_models = []
        
        for idx in range(self.n_bootstrap_samples):
            base_model_path = f"{filename}_model_{idx}.pkl"
            calibrated_model_path = f"{filename}_calibrated_{idx}.pkl"
            
            try:
                base_model = joblib.load(base_model_path)
                calibrated_model = joblib.load(calibrated_model_path)
            except FileNotFoundError as e:
                print(f"Error loading model {idx}: {e}")
                continue
            
            self.bootstrap_models.append(base_model)
            self.calibrated_models.append(calibrated_model)
        print(f"Models loaded from {filename}_model_*.pkl and {filename}_calibrated_*.pkl")

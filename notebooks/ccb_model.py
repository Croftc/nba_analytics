import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import joblib
from sklearn.utils import resample

class BootstrapCalibratedClassifier:
    def __init__(self, n_bootstrap_samples=8, base_model_params=None):
        self.n_bootstrap_samples = n_bootstrap_samples
        self.bootstrap_models = []
        self.calibrated_models = []
        self.base_model_params = base_model_params if base_model_params else {
            'tree_method': "hist",
            'enable_categorical': True,
            'verbosity': 1,
            'max_depth': 7,
            'learning_rate': 0.022,
            'n_estimators': 300,
            'gamma': 7,
            'subsample': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 15,
            'reg_lambda': 0.3,
            'reg_alpha': 0.3,
        }

    def fit(self, X_train, y_train):
        for _ in range(self.n_bootstrap_samples):
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

    def predict(self, X):
        # Aggregate predictions from bootstrap models
        preds = np.array([model.predict(X) for model in self.bootstrap_models])
        return np.round(np.mean(preds, axis=0)).astype(int)

    def predict_proba(self, X):
        # Aggregate calibrated predicted probabilities from bootstrap models
        proba = np.array([calibrated_model.predict_proba(X) for calibrated_model in self.calibrated_models])
        # Average probabilities across all bootstrap models
        return np.mean(proba, axis=0)

    def score(self, X_test, y_test):
        """
        This method calculates the accuracy score of the model.
        It compares the predicted labels with the true labels.
        """
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

    def save_model(self, filename):
        """Save the trained models to a file."""
        for idx, model in enumerate(self.bootstrap_models):
            model_path = f"{filename}_model_{idx}.pkl"
            joblib.dump(model, model_path)
        
        # Save calibrated models
        for idx, calibrated_model in enumerate(self.calibrated_models):
            calibrated_model_path = f"{filename}_calibrated_{idx}.pkl"
            joblib.dump(calibrated_model, calibrated_model_path)
        print(f"Models saved to {filename}")

    def load_model(self, filename):
        """Load both base and calibrated models."""
        self.bootstrap_models = []
        self.calibrated_models = []
        
        for idx in range(self.n_bootstrap_samples):
            base_model_path = f"{filename}_model_{idx}.pkl"
            calibrated_model_path = f"{filename}_calibrated_{idx}.pkl"
            
            base_model = joblib.load(base_model_path)
            calibrated_model = joblib.load(calibrated_model_path)
            
            self.bootstrap_models.append(base_model)
            self.calibrated_models.append(calibrated_model)
        print(f"Models loaded from {filename}")

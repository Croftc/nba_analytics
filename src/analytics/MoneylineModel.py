import xgboost as xgb
import os
import numpy as np
from analytics.Model import Model


class MoneylineModel(Model):
    def __init__(self, pretrained=True, base_model='xgb', do_ensemble=True):
        super().__init__()
        
        self.ensemble = []
        self.is_trained = pretrained
        self.do_ensemble = do_ensemble

        if pretrained:
            if base_model == 'xgb':
                print('loading model')
                self.model = xgb.XGBClassifier().load_model(self.model_file_loc)
                print('loading ensemble')

                if do_ensemble:
                    self.ensemble = self.__load_ensemble__()
                print('MODELS LOADED')
            else:
                print('unrecognized model type...')
                exit()
        else:
            if base_model == 'xgb':
                self.model = xgb.XGBClassifier()
            else:
                print('unrecognized model type...')
                exit()

        

    def __load_ensemble__(self):

        if len(self.ensemble) > 0:
            print('returning in memory ensemble')
            return self.ensemble
        for i, model in enumerate(os.listdir(self.ensemble_dir)):
            bst = xgb.XGBClassifier()
            bst.load_model(f'{self.ensemble_dir}{self.ensemble_base_model_name}_{i}.json')
            self.ensemble.append(bst)
        return self.ensemble
        

    def fit(self, X, y):
        self.base_model.fit(X, y)
        if self.do_ensemble:
            n_bootstrap_samples = 100  # Number of bootstrap samples
            bootstrap_models = []

            # Generate bootstrap samples and train a model on each
            # TODO: clean this up and parameterize stuff...
            for _ in range(n_bootstrap_samples):
                X_train_sample, y_train_sample = resample(X, y, n_samples=int(0.5*len(X)), replace=True)
                model = xgb.XGBClassifier(tree_method="hist", sampling_method='gradient_based', enable_categorical=True, device="cuda", verbosity=1, kwargs={'colsample_bytree': 0.7, 'gamma': 7, 'learning_rate': 0.02, 'max_depth': 12, 'min_child_weight': 8, 'n_estimators': 800, 'reg_alpha': 0.1, 'reg_lambda': 0.6, 'subsample': 0.8})
                model.fit(X_train_sample, y_train_sample)
                bootstrap_models.append(model)

    def predict(self, X):
        if self.do_ensemble and len(self.ensemble) > 0:
            preds = np.array([model.predict(X) for model in self.ensemble])
            values, counts = np.unique(preds, return_counts=True)
            ind = np.argmax(counts)
            return preds[ind]
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.do_ensemble and len(self.ensemble) > 0:
            preds = []
            for i, model in enumerate(self.ensemble):
                if i%100 == 0:
                    print(f'getting predictions from model {i}')
                preds.append(model.predict_proba(X))
            preds = np.array(preds)
            return preds.mean(axis=0).reshape(-1,2)
        else:
            return self.model.predict_proba(X)

    def save(self, path):
        self.model.save_model(self.model_file_loc)

    def load(self, path):
        self.model = xgb.XGBClassifier().load_model(self.model_file_loc)
        return self.model

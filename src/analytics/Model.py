class Model:
    def __init__(self, 
                    model_file='../models/best_model.json', 
                    ensemble_dir='../ensemble/', 
                    ensemble_base_model_name='e_model_1000'):

        self.model_file_loc = model_file
        self.ensemble_dir = ensemble_dir
        self.ensemble_base_model_name = ensemble_base_model_name

    def fit(self, X, y, ensemble=False):
        raise NotImplementedError("Subclass must implement abstract method")

    def predict(self, X, ensemble=False):
        raise NotImplementedError("Subclass must implement abstract method")

    def save(self, path):
        raise NotImplementedError("Subclass must implement abstract method")

    def load(self, path):
        raise NotImplementedError("Subclass must implement abstract method")

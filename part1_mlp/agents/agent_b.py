from pathlib import Path
from ..models import SimpleMLP

class Agent:
    def __init__(self):
        self.name = 'agent_b'

    def data(self):
        from ..utils import load_data
        return load_data()

    def model(self, X_train_shape):
        return SimpleMLP(input_dim=X_train_shape)

    def explain(self):
        return {'explanation': 'not implemented'}

    def run(self, output_root: Path):
        X_train, X_test, y_train, y_test, feature_names, class_names = self.data()
        model = self.model(X_train.shape[1])
        return {'agent': self.name, 'status': 'ready'}

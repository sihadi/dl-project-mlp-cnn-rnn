from pathlib import Path
from ..models import SimpleMLP

class Agent:
    """Simple redundant agent: does data, model, explain steps."""
    def __init__(self):
        self.name = 'agent_a'

    def data(self):
        from ..utils import load_data
        return load_data()

    def model(self, X_train_shape):
        return SimpleMLP(input_dim=X_train_shape)

    def explain(self):
        # placeholder for explainability (e.g., LIME)
        return {'explanation': 'not implemented'}

    def run(self, output_root: Path):
        X_train, X_test, y_train, y_test, feature_names, class_names = self.data()
        model = self.model(X_train.shape[1])
        # training omitted in agent template; prefer central train script
        return {'agent': self.name, 'status': 'ready'}

from pathlib import Path
from ..models import SimpleLSTM

class Agent:
    def __init__(self):
        self.name = 'agent_a'

    def data(self):
        from datasets import load_dataset
        ds = load_dataset('imdb')
        return ds

    def model(self):
        return SimpleLSTM(vocab_size=10000)

    def explain(self):
        return {'explanation': 'not implemented'}

    def run(self, output_root: Path):
        return {'agent': self.name, 'status': 'ready'}

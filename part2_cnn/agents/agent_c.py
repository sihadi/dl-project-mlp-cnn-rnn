from pathlib import Path
from ..models import SimpleCNN

class Agent:
    def __init__(self):
        self.name = 'agent_c'

    def data(self):
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        return train, test

    def model(self):
        return SimpleCNN()

    def explain(self):
        return {'explanation': 'not implemented'}

    def run(self, output_root: Path):
        return {'agent': self.name, 'status': 'ready'}

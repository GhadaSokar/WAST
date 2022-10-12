import torch
import torch.nn as nn

# an autoencoder model with a single hidden layer
class AE(nn.Module):
    def __init__(self, dataset, input_size, nhidden):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(input_size, nhidden, bias=True)
        self.fc2 = nn.Linear(nhidden, input_size, bias=True)
        self.input_size = input_size
        self.dataset = dataset 
    def forward(self, x):
        x0 = x.view(-1, self.input_size)
        x1 = torch.sigmoid(self.fc1(x0))
        x2 = self.fc2(x1)
        return x2 

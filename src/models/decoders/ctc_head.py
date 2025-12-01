import torch.nn as nn

class CTCHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        return self.log_softmax(self.fc(x))

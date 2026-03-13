import torch.nn as nn

class FakeImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 2)   # ✅ KEEP 768

    def forward(self, x):
        return self.fc(x)

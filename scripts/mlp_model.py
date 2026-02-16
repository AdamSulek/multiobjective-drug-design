import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features=1024, hidden_dim=512, num_hidden_layers=3, dropout_rate=0.2, out_features=2):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = self.backbone(x)
        return self.output(x)
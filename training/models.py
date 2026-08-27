import torch
import torch.nn as nn

class StrokeTCN(nn.Module):
    def __init__(self, in_dim=68, n_classes=8, channels=(64, 128, 128), k=5):
        super().__init__()
        layers, c_in = [], in_dim
        for i, c in enumerate(channels):
            d = 2 ** i
            layers += [nn.Conv1d(c_in, c, k, dilation=d, padding=d * (k - 1) // 2),
                       nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(0.2)]
            c_in = c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x):                    # (B,T,F)
        h = self.tcn(x.transpose(1, 2))      # (B,C,T)
        return self.head(h.mean(dim=2))      # (B,n_classes)

class RallyGRU(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):                    # (B,T,F)
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)      # (B,T) logits

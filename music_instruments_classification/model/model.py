import torch
import torch.nn as nn


class Block2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2)
            # nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            # nn.BatchNorm1d(out_ch),
            # nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class CNN2D(nn.Module):
    """Some Information about CNN2D"""

    def __init__(self, classes):
        super(CNN2D, self).__init__()
        self.conv1 = Block2D(1, 16)
        self.conv2 = Block2D(16, 32)
        self.conv3 = Block2D(32, 64)
        self.conv4 = Block2D(64, 128)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class LSTMModel(nn.Module):

    def __init__(self, classifier_output=2, feature_size=40, hidden_size=128,
                 num_layers=1, dropout=0.1, bidirectional=False, device='cpu'):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = 2 if bidirectional else 1
        self.device = device
        self.layer_norm = nn.LayerNorm(feature_size)
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            hidden_size * self.directions, classifier_output)

    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        return (torch.zeros(n * d, batch_size, hs).to(self.device),
                torch.zeros(n * d, batch_size, hs).to(self.device))

    def forward(self, x):
        # x.shape => seq_len, batch, feature
        x = self.layer_norm(x)
        hidden = self._init_hidden(x.size()[1])
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.classifier(hn[-1, ...])
        return out


# if __name__ == '__main__':
#     model = LSTMModel(bidirectional=True, num_layers=2)
#     batch = torch.rand((90, 1, 40))
#     output = model(batch)
#     print(output.shape)

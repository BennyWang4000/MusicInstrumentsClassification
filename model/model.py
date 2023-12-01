import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """
    params
    -----
        classes: int, number of classes
    """

    def __init__(self, classes: int):
        super(CNN2D, self).__init__()
        self.conv1 = Block2D(1, 16)
        self.conv2 = Block2D(16, 32)
        self.conv3 = Block2D(32, 64)
        self.conv4 = Block2D(64, 128)
        self.conv5 = Block2D(128, 256)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2176, 64)
        self.linear2 = nn.Linear(64, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        shape
        -----
            torch.Size([4, 1, 64, 1001])
            torch.Size([4, 1, 64, 1001])
            torch.Size([4, 16, 32, 500])
            torch.Size([4, 16, 32, 500])
            torch.Size([4, 32, 16, 250])
            torch.Size([4, 32, 16, 250])
            torch.Size([4, 64, 8, 125])
            torch.Size([4, 64, 8, 125])
            torch.Size([4, 128, 4, 62])
            torch.Size([4, 128, 4, 62])
            torch.Size([4, 31744])
        '''

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class Block2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=2, padding=1),
            # nn.BatchNorm2d(in_ch),
            # nn.LeakyReLU(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.25),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Block1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2)
            # nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            # nn.BatchNorm1d(out_ch),
            # nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class CNN1D(nn.Module):
    """
    params
    -----
        classes: int, number of classes
    """

    def __init__(self, classes: int):
        super(CNN1D, self).__init__()
        self.conv1 = Block1D(1, 16)
        self.conv2 = Block1D(16, 32)
        self.conv3 = Block1D(32, 64)
        self.conv4 = Block1D(64, 128)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        x = self.conv3(x)
        print(x.size())
        x = self.conv4(x)
        print(x.size())
        x = self.flatten(x)
        print(x.size())
        x = self.linear(x)
        print(x.size())
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

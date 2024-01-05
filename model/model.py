import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    """Some Information about Trans"""

    def __init__(self, classes=20, device='cpu'):
        super(TransformerClassifier, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, activation=nn.LeakyReLU(), device=device)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=6, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=128, nhead=8, activation=nn.LeakyReLU(), device=device)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=3)

        self.classifier = nn.Sequential(
            LiearClassifier(1280, 512, dropout=0.2, is_flatten=True),
            LiearClassifier(512, 256, dropout=0.2),
            LiearClassifier(256, 64, dropout=0.2),
            LiearClassifier(64, classes, dropout=0.2,
                            is_activate=False, is_sigmoid=False),
        )
        # self.classifier = nn.Sequential(
        #     LiearClassifier(128, 128, dropout=0.2),
        #     LiearClassifier(128, 128, dropout=0.2),
        #     LiearClassifier(128, 128, dropout=0.2),
        #     LiearClassifier(128, classes, dropout=0.2,
        #                     is_activate=False, is_sigmoid=False),
        # )

    def forward(self, x):
        x = self.transformer_encoder(x)
        # print('x', x.shape)
        # print('out', out.shape)
        # print('mel', mel.shape)
        # x = self.transformer_decoder(out, mel)
        # out = out.mean(1)
        x = self.classifier(x)
        return x


class LiearClassifier(nn.Module):
    '''
    Args:
        in_ch (int): 
        out_ch (int): 
        is_flatten (bool, optional): Defaults to False.
        dropout (int, optional): Defaults to 0.
    '''

    def __init__(self, in_ch: int, out_ch: int, is_flatten=False, dropout=0, is_activate=True, is_sigmoid=False):

        super(LiearClassifier, self).__init__()
        modules = []
        if is_flatten:
            modules.append(nn.Flatten())
        modules.extend([
            nn.Linear(in_ch, out_ch),
            nn.Dropout(dropout),
        ])
        if is_activate:
            modules.append(nn.LeakyReLU())
        if is_sigmoid:
            modules.append(nn.Softmax())

        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        return self.classifier(x)


class Trans(nn.Module):
    """Some Information about Trans"""

    def __init__(self):
        super(Trans, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=500, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=12, enable_nested_tensor=False)
        self.reduction = SVDReduction()
        self.classifier = Classifier(500, 20)
        self.conv = nn.Conv1d(128, 64, 3, 1, 2)
        self.attn_cnn = AttnCNN()

    def forward(self, x):
        x = x.squeeze(1)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = self.attn_cnn(x)
        return x


class SVDReduction(nn.Module):
    def __init__(self):
        super(SVDReduction, self).__init__()
        self.out_channels = 512

    def forward(self, x):
        U, S, _ = torch.linalg.svd(x)
        x = torch.mm(U[:, :self.out_channels],
                     torch.diag(S[:self.out_channels]))
        return x


class AttnCNN1D(nn.Module):
    def __init__(self, classes=20):
        super(AttnCNN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(10, 20, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            nn.Conv1d(20, 20, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 40, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(40, 40, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(40, 80, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(80, 80, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, classes),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.attn1 = SpatialAttn1D(
            in_features=80, normalize_attn=True)
        self.attn2 = SpatialAttn1D(
            in_features=80, normalize_attn=True)
        self.attn3 = SpatialAttn1D(
            in_features=80, normalize_attn=True)
        self.projector1 = ProjectorBlock(20, 80)
        self.projector2 = ProjectorBlock(40, 80)

        # # extract weights from `vggish_list`
        # for seq in (self.features, self.embeddings):
        #     for layer in seq:
        #         if type(layer).__name__ != "MaxPool2d" and type(layer).__name__ != "ReLU":
        #             layer.weight = next(param_generator())
        #             layer.bias = next(param_generator())

    def forward(self, x):
        print('x', x.shape)
        l1 = self.conv1(x)
        print('l1', l1.shape)
        l2 = self.conv2(l1)
        print('l2', l2.shape)
        l3 = self.conv3(l2)
        print('l3', l3.shape)
        # l4 = self.conv4(l3)
        # l4 = self.dense(l3)
        # print('l4', l4.shape)
        # print('p1', self.projector1(l1).shape)
        # print('p2', self.projector2(l2).shape)
        c1, g1 = self.attn1(self.projector1(l1), l3)
        c2, g2 = self.attn1(self.projector2(l2), l3)
        g = torch.cat((g1, g2), dim=1)
        x = self.classifier(g)
        return x


class AttnCNN(nn.Module):
    def __init__(self, classes=20):
        super(AttnCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1984, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, classes),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.attn1 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.attn2 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.attn3 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.projector1 = ProjectorBlock(64, 256)
        self.projector2 = ProjectorBlock(128, 256)

        # # extract weights from `vggish_list`
        # for seq in (self.features, self.embeddings):
        #     for layer in seq:
        #         if type(layer).__name__ != "MaxPool2d" and type(layer).__name__ != "ReLU":
        #             layer.weight = next(param_generator())
        #             layer.bias = next(param_generator())

    def forward(self, x):
        print('x', x.shape)
        l1 = self.conv1(x)
        print('l1', l1.shape)
        l2 = self.conv2(l1)
        print('l2', l2.shape)
        l3 = self.conv3(l2)
        print('l3', l3.shape)
        # l4 = self.conv4(l3)
        # l4 = self.dense(l3)
        # print('l4', l4.shape)
        # print('p1', self.projector1(l1).shape)
        # print('p2', self.projector2(l2).shape)
        c1, g1 = self.attn1(self.projector1(l1), l3)
        c2, g2 = self.attn1(self.projector2(l2), l3)
        g = torch.cat((g1, g2), dim=1)
        x = self.classifier(g)
        return x


class VGGAttnEX(nn.Module):
    def __init__(self, classes=20):
        super(VGGAttnEX, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )
        self.embeddings = nn.Sequential(
            nn.Linear(512*24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*3, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, classes),
            nn.Sigmoid(),
        )
        self.attn1 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.attn2 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.attn3 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.projector1 = ProjectorBlock(128, 512)
        self.projector2 = ProjectorBlock(256, 512)
        self.dense = nn.Sequential(
            nn.Conv2d(512, 512, (2, 10), 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (1, 10), 2),
            nn.ReLU(inplace=True),
        )

        # # extract weights from `vggish_list`
        # for seq in (self.features, self.embeddings):
        #     for layer in seq:
        #         if type(layer).__name__ != "MaxPool2d" and type(layer).__name__ != "ReLU":
        #             layer.weight = next(param_generator())
        #             layer.bias = next(param_generator())

    def forward(self, x):
        l1 = self.conv1(x)
        # print('l1', l1.shape)
        l2 = self.conv2(l1)
        # print('l2', l2.shape)
        l3 = self.conv3(l2)
        # print('l3', l3.shape)
        l4 = self.conv4(l3)
        # print('l4', l4.shape)
        l4 = self.dense(l4)
        # print('l4', l4.shape)
        # print('p1', self.projector1(l1).shape)
        # print('p2', self.projector2(l2).shape)
        c1, g1 = self.attn1(self.projector1(l1), l4)
        c2, g2 = self.attn1(self.projector2(l2), l4)
        c3, g3 = self.attn1(l3, l4)
        g = torch.cat((g1, g2, g3), dim=1)
        x = self.classifier(g)
        return x, c1, c2, c3


class CNN2D(nn.Module):
    """
    params
    -----
        classes: int, number of classes
    """

    def __init__(self, classes: int):
        super(CNN2D, self).__init__()
        self.conv1 = Block2D(1, 64)
        self.conv2 = Block2D(64, 128)
        self.conv3 = Block2D(128, 256)
        self.conv4 = Block2D(256, 512)
        self.conv5 = Block2D(512, 512)
        # self.conv6 = Block2D(512, 512)
        self.dense = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = Classifier(1024, classes)
        self.projector = ProjectorBlock(256, 512)
        self.attn1 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.attn2 = SpatialAttn(
            in_features=512, normalize_attn=True)
        self.attn2 = SpatialAttn(
            in_features=512, normalize_attn=True)

    def forward(self, x):
        '''
        shape
        -----
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        l1 = self.conv3(x)
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)
        l2 = self.conv4(x)
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)
        l3 = self.conv5(x)
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)
        # print(x.shape)
        # x = self.conv6(x)
        # x = self.pool(x)
        x = self.dense(x)
        print(x.shape)
        print(l2.shape)
        print(l3.shape)
        c1, g1 = self.attn1(self.projector(l1), x)
        c2, g2 = self.attn2(l2, x)
        c3, g3 = self.attn3(l3, x)

        g = torch.cat((g1, g2, g3), dim=1)  # batch_sizex3C
        # classification layer
        x = self.classify(g)  # batch_sizexnum_classes
        return [x, c1, c2, c3]


class Classifier(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_ch),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.linear(x)


class Block2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=2, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.op(x)
        return x


class SpatialAttn1D(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn1D, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv1d(in_channels=in_features, out_channels=1,
                            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, H, W = l.size()
        # N, C, H, W = l.size()
        c = self.op(l+g)  # (batch_size,1,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=1).view(N, H, W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, -1).sum(dim=2)  # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N)
        return c.view(N, 1, H, W), g


class SpatialAttn(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1,
                            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        # N, H, W = l.size()
        N, C, H, W = l.size()
        c = self.op(l+g)  # (batch_size,1,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, -1).sum(dim=2)  # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N)
        return c.view(N, 1, H, W), g


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

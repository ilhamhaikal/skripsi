import torch.nn as nn

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, nClass):
        super(CRNN, self).__init__()

        kernelSize = [3, 3, 3, 3, 3, 3, 2]
        paddingSize = [1, 1, 1, 1, 1, 1, 0]
        strideSize = [1, 1, 1, 1, 1, 1, 1]
        nMaps = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv(i, batchNormalization=False):
            channelIn = 1 if i == 0 else nMaps[i - 1]
            channelOut = nMaps[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(channelIn, channelOut, kernelSize[i], strideSize[i], paddingSize[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(channelOut))

            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # CNN Layer
        conv(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        conv(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv(2, True)
        conv(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv(4, True)
        conv(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv(6, True)  # 512x1x16

        self.cnn = cnn
        # self.cnn = nn.Sequential(ConvolutionalNN)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, nClass))

    def forward(self, input):
        # CNN
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [width, batch, feature]
        # print(conv.size())

        # RNN
        output = self.rnn(conv)

        return output



# import torch
# model = CRNN(37)
# image = torch.FloatTensor(64, 1, 32, 120)
# model(image)
# print(model)


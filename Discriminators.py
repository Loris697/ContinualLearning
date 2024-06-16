import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class DiscriminatorConv(nn.Module):
    def __init__(self, ngpu = 1, nc=1, ndf=64):
        super(DiscriminatorConv, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class DiscriminatorB0(nn.Module):
    # initializers
    def __init__(self, in_channels = 1):
        super(DiscriminatorB0, self).__init__()
        self.classifier = efficientnet_b0()

        #changing input features to handle grayscale image
        #get old parameters
        out_channels = self.classifier.features[0][0].out_channels
        stride = self.classifier.features[0][0].stride
        padding = self.classifier.features[0][0].padding
        bias = self.classifier.features[0][0].bias
        kernel_size = self.classifier.features[0][0].kernel_size

        self.classifier.features[0][0] = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding, 
            bias = bias
        )

        #changing last layer to adapt for the binary task
        fc_in_features = self.classifier.classifier[1].in_features
        self.classifier.classifier[1] = nn.Linear(fc_in_features, 1)
        self.classifier  = self.classifier.cuda()

    # forward method
    def forward(self, input):
        return self.classifier(input)
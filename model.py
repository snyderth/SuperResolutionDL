import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules import upsampling

# torch.nn.Conv2d(  in_channels, 
#                   out_channels, 
#                   kernel_size, 
#                   stride=1, 
#                   padding=0, 
#                   dilation=1, 
#                   groups=1, 
#                   bias=True, 
#                   padding_mode='zeros', 
#                   device=None, 
#                   dtype=None)
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight)


class Net1(nn.Module):
    def __init__(self, upscale_factor):
        super(Net1, self).__init__()
        # Define layers
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv3 = nn.Conv2d(64, upscale_factor**6, (3, 3), (1,1), (1,1))
        self.conv4 = nn.Conv2d(16, 32, (3, 3), (1,1), (1,1))
        self.conv5 = nn.Conv2d(32, upscale_factor**4, (3,3),(1,1), (1,1))
        self.maxpool = nn.MaxPool2d((4,4))
        self._initialize_weights()


    def forward(self, x):
        # assign layers
        # print(x.shape)
        x = (self.relu(self.conv1(x)))
        # print(x.shape)
        x = self.maxpool(self.relu(self.conv2(x)))
        # print(x.shape)
        x = (self.pixel_shuffle(self.conv3(x)))
        x = self.pixel_shuffle(x)
        x = self.pixel_shuffle(x)
        # print(x.shape)
        # x = (self.relu(self.conv4(x)))
        # print(x.shape)
        # x = self.pixel_shuffle(self.conv5(x))
        # print(x.shape)
        # x = self.pixel_shuffle(x)
        # print(x.shape)
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv5.weight)

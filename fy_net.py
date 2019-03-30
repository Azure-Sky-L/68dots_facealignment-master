from torch import nn
import torch
from torch.nn import functional as F

def conv3_3(in_planes, out_planes, strd = 1, padding = 1, bias = False):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3,
                stride = strd, padding = padding, bias = bias)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, need_shortcut = False):
        super(ConvBlock, self).__init__()
#        self.conv = conv3_3(in_channels, int(out_channels) / 4)
        self.conv = nn.Conv2d(in_channels, int(out_channels) / 4, kernel_size = 1, stride = 1, padding = 0)
        self.bn = nn.BatchNorm2d(int(out_channels) / 4)
        self.conv0 = conv3_3(int(out_channels) / 4, int(out_channels) / 2)
        self.bn0 = nn.BatchNorm2d(int(out_channels) / 2)
        self.conv1 = conv3_3(int(out_channels) / 2, int(out_channels) / 4)
        self.bn1 = nn.BatchNorm2d(int(out_channels) / 4)
        self.conv2 = conv3_3(int(out_channels) / 4, int(out_channels) / 4)
        self.bn2 = nn.BatchNorm2d(int(out_channels) / 4)
        
        if need_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True)
            )
        else:
            self.shortcut = None
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out, True)
        out = self.conv0(out)
        out = self.bn0(out)
        out = F.relu(out, True)
        
        out1 = self.conv1(out)
        out1 = self.bn1(out1)
        out1 = F.relu(out1, True)
        
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = F.relu(out2, True)
        
        out2 = torch.cat((out, out1, out2), 1)
        
        residual = x if self.shortcut is None else self.shortcut(x)
        
        out2 += residual
        
        return out2
    
class FAN(nn.Module):
    
    def __init__(self, num_modules = 2):
        super(FAN, self).__init__()
        self.num_modules = num_modules
    
        self.conv = nn.Conv2d(1, 16, kernel_size = 3, stride = 2, padding = 1)
        self.bn = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 16, kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(16, 64, kernel_size = 1, stride = 2, padding = 0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv12 = nn.Conv2d(128, 68, kernel_size = 1, stride = 1, padding = 0)
        self.layer1 = ConvBlock(64, 64, need_shortcut = True)
        self.layer2 = ConvBlock(64, 64, need_shortcut = False)
        self.conv0 = nn.Conv2d(16, 64, kernel_size = 1, stride = 1, padding = 0) 
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1)
#        self.pool = nn.MaxPool2d(2, stride=2, return_indices = True)
#        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.pool = nn.AvgPool2d(2, stride = 2)
        self.pixel = nn.PixelShuffle(2)
#        self.upsample = nn.Upsample(scale_factor=2, mode= 'bilinear' )
        self.fc = nn.Linear(4096, 136) #

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        
        out1 = F.relu(out, True)
        
        out2 = self.conv1(out1)
        out2 = self.bn1(out2)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out2 = self.bn2(out2)
        out2 = F.relu(out2, True)
        
        out2 = torch.cat((out2, out2), 1)
        
        out3 = self.conv3(out)
        out3 = self.bn3(out3)
        out3 = F.relu(out3, True)
        
        out3 += out2
        
        out4 = self.layer1(out3)
        # out5 = F.relu(out3)
        out5 = self.pool(out3)
        out5 = F.relu(out5)
        out5 = self.layer1(out5)
        
        out6 = self.layer2(out5)
        # out7 = F.relu(out5)
        out7 = self.pool(out5)
        out7 = F.relu(out7)
        out7 = self.layer2(out7)        
        
        out8 = self.layer2(out7)
#        out9 = self.fc(self.conv4(out8).view(-1, 4096))
        out9 = self.fc(out8.view(-1, 4096))
        out10 = self.pixel(out8)
        out10 = self.conv0(out10)
#        out10 = self.unpool(out8, idd0)
#        out10 = self.upsample(out8)
        out10 += out6
        
        out11 = self.layer2(out10)
        out11 = self.pixel(out11)
        out11 = self.conv0(out11)
#        out11 = self.unpool(out11, idd)
#        out11 = self.upsample(out10)
        out11 += out4
        
        out12 = self.layer2(out11)
        out12 = self.conv11(out12)
        out12 = self.conv12(out12)
        heat_map = out12
        out12 = F.sigmoid(out12)
#        print(out12.size())
#        out12 = out12.view(16, 68, 1, 4096)
#        out12 = F.softmax(out12)

        outputs = []
        reg_outputs = []
        outputs.append(heat_map)
        reg_outputs.append(out9)
#        return heat_map, out9
        return outputs, reg_outputs

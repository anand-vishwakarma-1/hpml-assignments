import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_, out, stride=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_, out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out, out, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ != self.expansion*out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_, self.expansion*out,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet, self).__init__()
        block, num_blocks = ResnetBlock, [2, 2, 2, 2]
        self.in_ = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_, out, stride))
            self.in_ = out * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

import torch
import torch.nn as nn

# The model should perform better when layers are increased
# in reality it worsens the performance but
# ResNet has skip connection that send output to layer after it and also 
# the next ones, in this way the model remembers what it learned before
# and in theory it never becomes worse

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.identity_downsample = identity_downsample
        self.expansion = 4  # In each layer the channels leaving is 4 times of channels entering
        # 1 x 1, x
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False,)
        self.norm1 = nn.BatchNorm2d(intermediate_channels)
        # 3 x 3, x
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False,)
        self.norm2 = nn.BatchNorm2d(intermediate_channels)
        # 1 x 1, x * 4
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False,)
        self.norm3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        if self.identity_downsample is not None:  # Adding skip connection by changing shape of previous x
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):  # Layers in resnet50 - [3, 4, 6, 3]
    def __init__(self, block, layers, img_channels, n_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.in_channels = 64

        self.layer1 = self.make_layer(block, layers[0], 64, 1)
        self.layer2 = self.make_layer(block, layers[1], 128, 2)
        self.layer3 = self.make_layer(block, layers[2], 256, 2)
        self.layer4 = self.make_layer(block, layers[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512*4, n_classes)

    def make_layer(self, block, n_times_block, intermediate_channels, stride):  # One of the layers are going to have stride of 2
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            # if identity (skip connection) in is not the quarter or of what it will be in future
            # Between channels leaving 1st layer is 256 but input to 2nd is 128 *******
            # so we add an extra layer to make the shapes match where identity can be added to x later
            # Therefore it is always add only to first block
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,
                                                          intermediate_channels * 4,
                                                          kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(intermediate_channels * 4))

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        # till now the in_channel is what it enters, the layer, but it expands with 4
        # and stays same entire time until next block in same layer
        self.in_channels = intermediate_channels * 4

        for i in range(n_times_block - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_out(x)
        return x

def ResNet50(img_channels=3, n_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, n_classes)

# def ResNet101(img_channels=3, n_classes=1000):
#     return ResNet(block, [3, 4, 23, 3], img_channels, n_classes)
#
# def ResNet152(img_channels=3, n_classes=1000):
#     return ResNet(block, [3, 8, 36, 3], img_channels, n_classes)


model = ResNet50()
x = torch.randn(2, 3, 224, 224)
print(model(x).shape)
import torch.nn as nn
from check_params import compare_model_params


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample = None,
        stride = 1,
        expansion = 4,
    ):
        super().__init__()
        expanded_channels = out_channels * expansion 
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels = out_channels, out_channels = expanded_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(expanded_channels)
        self.act3 = nn.ReLU()

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        if self.downsample:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x += identity
        x = self.relu(x)

        return x


class Resnet(nn.Module):
    def __init__(self, layers, image_channels, num_classes):
        super().__init__()
        
        self.in_channels = 64

        conv1_padding = round((111 * 2 - 224 + 7) / 2)
        maxpool_padding = round((55 * 2 - 112 + 4) / 2)

        self.conv1 = nn.Conv2d(in_channels = image_channels, out_channels = self.in_channels, kernel_size = 7, stride = 2, padding = conv1_padding, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = maxpool_padding)
        
        self.layer1 = self.make_resnet_layers(num_res_layers = layers[0], out_channels = 64, downsample_stride = 1, bottleneck_stride = 1)
        self.layer2 = self.make_resnet_layers(num_res_layers = layers[1], out_channels = 128, downsample_stride = 2, bottleneck_stride = 2)
        self.layer3 = self.make_resnet_layers(num_res_layers = layers[2], out_channels = 256, downsample_stride = 2, bottleneck_stride = 2)
        self.layer4 = self.make_resnet_layers(num_res_layers = layers[3], out_channels = 512, downsample_stride = 2, bottleneck_stride = 2)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size = (1, 1))

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(2048, num_classes)

    def make_resnet_layers(
        self,
        num_res_layers,
        out_channels,
        downsample_stride,
        bottleneck_stride,
    ):
        layers = []

        downsample = self.downsample_layer(in_channels = self.in_channels, expanded_channels = out_channels * 4, stride = downsample_stride)
        
        layers.append(Bottleneck(in_channels = self.in_channels, out_channels = out_channels, downsample = downsample, stride = bottleneck_stride))

        self.in_channels = out_channels * 4

        for i in range(num_res_layers - 1):

            layers.append(Bottleneck(in_channels = self.in_channels, out_channels = out_channels))
        
        return nn.Sequential(*layers)


    @staticmethod
    def downsample_layer(in_channels, expanded_channels, stride):

        downsample_lay = nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = expanded_channels, kernel_size = 1, stride = stride, bias = False), 
                                       nn.BatchNorm2d(expanded_channels))
        
        return downsample_lay

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


## Chequea que su implementación sea correcta. No borrar!!

model = Resnet(layers = [3, 4, 6, 3], image_channels = 3, num_classes = 1000)
compare_model_params(model)

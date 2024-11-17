import torch.nn as nn
from check_params import compare_model_params


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=None,
        stride=1,
        expansion=4,
    ):
        super().__init__()
        expanded_channels = out_channels * expansion
        ############ Cree su código a continuación ##################################

    def forward(self, x):
        identity = x

        ############ Cree su código a continuación ##################################
        pass


class Resnet(nn.Module):
    def __init__(self, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64

        ############ Cree su código a continuación ##################################
        pass

    def make_resnet_layers(
        self,
        num_res_layers,
        out_channels,
        downsample_stride,
        bottleneck_stride,
    ):
        layers = []

        ############ Cree su código a continuación ##################################
        pass

    @staticmethod
    def downsample_layer(in_channels, expanded_channels, stride):
        ############ Cree su código a continuación ##################################
        pass

    def forward(self, x):
        ############ Cree su código a continuación ##################################
        pass


## Chequea que su implementación sea correcta. No borrar!!

model = Resnet(layers=[3, 4, 6, 3], image_channels=3, num_classes=1000)
compare_model_params(model)

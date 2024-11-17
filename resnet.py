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
        # Definimos los bloques convolucionales usando bottleneck_block
        self.bottleneck_blocks = nn.Sequential(
            self.bottleneck_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0),  # 1x1 Conv
            self.bottleneck_block(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),  # 3x3 Conv
            self.bottleneck_block(out_channels, expanded_channels, kernel_size=1, stride=1, padding=0),  # 1x1 Conv
        )

        # Downsample para ajustar la identidad
        self.downsample = downsample

        # ReLU final
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        ############ Cree su código a continuación ##################################
        identity = x

        # Paso por capas usando ModuleDict
        # Paso por los bloques del Bottleneck
        out = self.bottleneck_blocks(x)

        # Si es necesario ajustar la identidad con downsample
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection y activación final
        out += identity
        out = self.relu(out)

        return out
        pass

    @staticmethod
    def bottleneck_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        """
        Definimos un bloque reutilizable que combina convolución, BatchNorm y activación ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Resnet(nn.Module):
    def __init__(self, layers, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64

        # Primera parte de la red (conv1, bn1, relu, maxpool)
        self.initial_block = nn.Sequential(
            nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

         # Creación de bloques ResNet usando ModuleDict
        self.res_blocks = nn.ModuleDict({
            "layer1": self.make_resnet_layers(layers[0], 64, downsample_stride=1,bottleneck_stride=1),
            "layer2": self.make_resnet_layers(layers[1], 128, downsample_stride=2,bottleneck_stride=1),
            "layer3": self.make_resnet_layers(layers[2], 256, downsample_stride=2,bottleneck_stride=1),
            "layer4": self.make_resnet_layers(layers[3], 512, downsample_stride=2,bottleneck_stride=1),
        })


        # Average Pool y Fully Connected
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 4, num_classes)
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
        # Crear el downsample para el primer bloque si es necesario
        downsample = None
        if downsample_stride != 1 or self.in_channels != out_channels * 4:
            downsample = self.downsample_layer(self.in_channels, out_channels * 4, stride=downsample_stride)

        # Primer bloque Bottleneck con downsample
        layers.append(
            Bottleneck(
                in_channels=self.in_channels,
                out_channels=out_channels,
                downsample=downsample,
                stride=downsample_stride
            )
        )

        # Actualizar self.in_channels para los siguientes bloques
        self.in_channels = out_channels * 4

        # Añadir los bloques restantes (sin downsample)
        for _ in range(1, num_res_layers):
            layers.append(
                Bottleneck(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1  # Stride=1 porque no hay más reducción de tamaño después del primer bloque
                )
            )

        # Retornar un bloque Sequential con todas las capas creadas
        return nn.Sequential(*layers)
        pass

    @staticmethod
    def downsample_layer(in_channels, expanded_channels, stride):
        ############ Cree su código a continuación ##################################
        """
        Crea un bloque para ajustar los canales en la conexión residual (skip connection).
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(expanded_channels),
        )
        pass

    def forward(self, x):
        ############ Cree su código a continuación ##################################
        """
        Realiza el forward pass en el modelo Resnet.
        """
        # Primera parte: Bloque inicial (conv1, bn1, relu, maxpool)
        x = self.initial_block(x)

        # Pasar por cada conjunto de bloques (layer1 a layer4) de self.res_blocks
        for layer in self.res_blocks.values():  
            x = layer(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten y Fully Connected
        x = self.flatten(x)
        x = self.fc(x)

        return x

        pass


## Chequea que su implementación sea correcta. No borrar!!

model = Resnet(layers=[3, 4, 6, 3], image_channels=3, num_classes=1000)
compare_model_params(model)

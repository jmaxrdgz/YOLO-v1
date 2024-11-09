import torch
import torch.nn as nn

'''
Architecture configuration of the conv layers part of the network
Also called "darknet"
'''
darknet_config = [
    # kernel_size, filters, stride, padding
    (7, 64, 2, 7),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # Last index is number of repetition
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
    
class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.conv(x))
    
class YOLO_v1(nn.Module):
    '''
    24 convolutional layers (defined in darknet_config) followed by 2 fully connected layers
    '''
    def __init__(self, in_channels=3, **kwargs):
        super(YOLO_v1, self).__init__()
        self.in_channels = in_channels
        self.darknet_config = darknet_config
        self.darknet = self._create_layers(self.darknet_config)
        self.fully_connected = self._create_fully_connected(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fully_connected(torch.flatten(x, start_dim=1))
    
    def _create_layers(self, config):
        layers = []
        in_channels = self.in_channels

        for layer in config:
            if isinstance(layer, tuple):
                kernel_size, filters, stride, padding = layer
                layers.append(CNN_Block(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding))
                in_channels = filters

            elif isinstance(layer, str) and layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif isinstance(layer, list):
                for _ in range(layer[-1]):
                    for sub_layer in layer[:-1]:
                        kernel_size, filters, stride, padding = sub_layer
                        layers.append(CNN_Block(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding))
                        in_channels = filters

        return nn.Sequential(*layers)
    
    def _create_fully_connected(self, split_size, num_boxes, num_classes):
        '''
        Image is divided into an S x S
        each grid cell predicts B bounding boxes, confidence for those boxes
        C class probabilities
        encoded as an S x S x (B x 5 + C) tensor.
        '''
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(S * S * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C)), #
            nn.LeakyReLU(0.1)
        )

def test(self): 
    model = YOLO_v1(split_size=7, num_boxes=2, num_classes=20)
    x = torch.rand((2, 3, 448, 448))
    print(model(x).shape) # Outputs torch.Size([2, 1470]) : 7x7x30 = 1470 so we are good !
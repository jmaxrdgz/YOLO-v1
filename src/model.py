import torch

class YOLOModel(torch.nn.module):

    def __init__(self):
        super(YOLOModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, 7)
        self.maxPool1 = torch.nn.MaxPool2d(112, 2)

        self.conv2 = torch.nn.Conv2d(64, 192, 3)
        self.maxPool2 = torch.nn.MaxPool2d(56, 2)

        self.conv3 = torch.nn.Conv2d(192, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3)
        self.conv5 = torch.nn.Conv2d(256, 256, 1)
        self.conv6 = torch.nn.Conv2d(256, 512, 3)
        self.maxPool3 = torch.nn.MaxPool2d(28, 2)

        self.conv7 = torch.nn.Conv2d(512, 256, 1)
        self.conv8 = torch.nn.Conv2d(256, 512, 3)
        self.conv9 = torch.nn.Conv2d(512, 512, 1)
        self.conv10 = torch.nn.Conv2d(512, 1024, 3)
        self.maxPool4 = torch.nn.MaxPool2d(14, 2)

        self.conv11 = torch.nn.Conv2d(1024, 512, 1)
        self.conv12 = torch.nn.Conv2d(512, 1024, 3)
        self.conv13 = torch.nn.Conv2d(1024, 1024, 3)
        self.maxPool5 = torch.nn.MaxPool2d(7, 3)

        self.conv14 = torch.nn.Conv2d(1024, 1024, 3)
        self.conv15 = torch.nn.Conv2d(1024, 1024, 3)

        self.connected1 = torch.nn.Linear(1024, 4096)
        self.activation1 = torch.nn.ReLU()

        self.connected2 = torch.nn.Linear(4096, 7 * 7 * 30)
        self.ectivation2 = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool1(x)

        x = self.conv2(x)
        x = self.maxPool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxPool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxPool4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxPool5(x)

        x = self.conv15(x)
        x = self.conv14(x)

        x = self.connected1(x)
        x = self.activation1(x)

        x = self.connected2(x)
        x = self.ectivation2(x)
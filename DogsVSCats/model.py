import torch
from torch import nn
from torchvision.models import squeezenet1_1


class SqueezeNet(nn.Module):
    """

    """

    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.net = squeezenet1_1(pretrained=True)  # Load the pretrained model.
        self.net.num_classes = 2

        # Finetune the squeezenet to 2 class.
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, self.net.num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    def forward(self, x):
        return self.net(x)

import torch
from torch import nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(input_c, num_filters, kernel_size=(4,4), stride=(2,2), padding=(1,1)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(num_filters, num_filters*2, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
                nn.BatchNorm2d(num_filters*2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(num_filters*2, num_filters*4, kernel_size=(4,4), stride=(2,2), padding=(1,1), bias=False),
                nn.BatchNorm2d(num_filters*4),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(num_filters*4, num_filters*8, kernel_size=(4,4), stride=(1,1), padding=(1,1), bias=False),
                nn.BatchNorm2d(num_filters*8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(num_filters*8, 1, kernel_size=(4,4), stride=(1,1), padding=(1,1))
            )
        )

    def forward(self, x):
      return self.model(x)
import torch


class Net(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=3),
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7,stride= 1,padding= 0),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=3),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=3),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7,stride= 1, padding=0),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding=3),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1,padding= 0),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(inplace=True)
        )

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, data:torch.Tensor)->torch.Tensor:
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data = self.conv5(data)
        data = self.final(data)
        return data

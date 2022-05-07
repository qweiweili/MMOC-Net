import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Norm_Activation(nn.Module):
    def __init__(self, feat_in, feat_out, kernel_size=3, stride=2, padding=1, bias=False, dilation=1, Norm=nn.BatchNorm2d, Activation=nn.ReLU):
        super(Conv_Norm_Activation, self).__init__()
        self.CNA = nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
            Norm(feat_out),
            Activation())

    def forward(self, x):
        x = self.CNA(x)
        return x

class Res_Block(nn.Module):
    def __init__(self, channels, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Res_Block, self).__init__()
        self.res = nn.Sequential(
            norm_layer(channels),
            activation_layer(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(channels),
            activation_layer(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        r = self.res(x)
        x = x + r
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return x + self.project(res)

class Full_Resolution_Net(nn.Module):
    def __init__(self, in_channel, classes=1):
        super(Full_Resolution_Net, self).__init__()
        self.in_conv = Conv_Norm_Activation(in_channel, 32, 3, 1, 1, Norm=nn.BatchNorm2d, Activation=nn.ReLU)
        self.res_block1 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block2 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block3 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.ASPP_block = ASPP(32, [4, 8, 12])
        self.res_block4 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block5 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block6 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.out = nn.Conv2d(32, classes, 1, 1, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.ASPP_block(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.out(x)
        return x

class Full_Resolution_Net_without_ASPP(nn.Module):
    def __init__(self, in_channel, classes=1):
        super(Full_Resolution_Net_without_ASPP, self).__init__()
        self.in_conv = Conv_Norm_Activation(in_channel, 32, 3, 1, 1, Norm=nn.BatchNorm2d, Activation=nn.ReLU)
        self.res_block1 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block2 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block3 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block4 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block5 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.res_block6 = Res_Block(32, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.out = nn.Conv2d(32, classes, 1, 1, 0)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    net = Full_Resolution_Net(1, 1)
    a = torch.rand((2, 1, 16, 16))
    b = net(a)
    print(b.shape)
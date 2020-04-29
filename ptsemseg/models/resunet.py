import torch.nn as nn
import torch
from ptsemseg.models.utils import unetUp

class resunet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(resunet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        # filters = [128,256,512,1024,2048]
        filters=[256,512,1024,2048]
        # filters=[128,256,512,1024]
        # filters=[64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]
        print(filters)

        # downsampling


        self.conv1 = nn.Sequential( nn.Conv2d(in_channels, filters[0], 3, 1, 1),  nn.BatchNorm2d(filters[0]), nn.ReLU())
        self.conv1_1=nn.Conv2d(filters[0], filters[0],3, 1, 1)
        self.shortcut1 =nn.Sequential(nn.Conv2d(self.in_channels, filters[0], 1, 1, 0),nn.BatchNorm2d(filters[0]))


        self.conv2 = nn.Sequential( nn.BatchNorm2d(filters[0]), nn.ReLU(),nn.Conv2d(filters[0], filters[1], 3, 2, 1))#size/2
        self.conv2_2=nn.Sequential( nn.BatchNorm2d(filters[1]), nn.ReLU(),nn.Conv2d(filters[1], filters[1], 3, 1, 1))#hold size
        self.shortcut2 = nn.Sequential(nn.Conv2d(filters[0], filters[1], 1, 2, 0),nn.BatchNorm2d(filters[1]))

        self.conv3 = nn.Sequential( nn.BatchNorm2d(filters[1]), nn.ReLU(),nn.Conv2d(filters[1], filters[2], 3, 2, 1))#size/2
        self.conv3_3 = nn.Sequential(nn.BatchNorm2d(filters[2]), nn.ReLU(),nn.Conv2d(filters[2], filters[2], 3, 1, 1))  # hold size
        self.shortcut3 = nn.Sequential(nn.Conv2d(filters[1], filters[2], 1, 2, 0),nn.BatchNorm2d(filters[2]))


        self.center = nn.Sequential( nn.BatchNorm2d(filters[2]), nn.ReLU(),nn.Conv2d(filters[2], filters[3], 3, 2, 1))#size/2
        self.center_1= nn.Sequential(nn.BatchNorm2d(filters[3]), nn.ReLU(),nn.Conv2d(filters[3], filters[3], 3, 1, 1))  # hold size
        self.shortcut_center = nn.Sequential(nn.Conv2d(filters[2], filters[3], 1, 2, 0),nn.BatchNorm2d(filters[3]))

        # upsampling
        self.up_sampling3=nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.up_conv3=nn.Sequential(nn.BatchNorm2d(filters[3]), nn.ReLU(),nn.Conv2d(filters[3], filters[2], 3, 1, 1))
        self.up_conv3_3 = nn.Sequential(nn.BatchNorm2d(filters[2]), nn.ReLU(), nn.Conv2d(filters[2], filters[2], 3, 1, 1))
        self.up_shortcut3=nn.Sequential(nn.Conv2d(filters[3], filters[2], 1, 1, 0),nn.BatchNorm2d(filters[2]))

        self.up_sampling2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up_conv2 = nn.Sequential(nn.BatchNorm2d(filters[2]), nn.ReLU(), nn.Conv2d(filters[2], filters[1], 3, 1, 1))
        self.up_conv2_2 = nn.Sequential(nn.BatchNorm2d(filters[1]), nn.ReLU(),nn.Conv2d(filters[1], filters[1], 3, 1, 1))
        self.up_shortcut2 = nn.Sequential(nn.Conv2d(filters[2], filters[1], 1, 1, 0),nn.BatchNorm2d(filters[1]))

        self.up_sampling1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up_conv1 = nn.Sequential(nn.BatchNorm2d(filters[1]), nn.ReLU(), nn.Conv2d(filters[1], filters[0], 3, 1, 1))
        self.up_conv1_1 = nn.Sequential(nn.BatchNorm2d(filters[0]), nn.ReLU(),nn.Conv2d(filters[0], filters[0], 3, 1, 1))
        self.up_shortcut1 =nn.Sequential( nn.Conv2d(filters[1], filters[0], 1, 1, 0),nn.BatchNorm2d(filters[0]))

        # final conv (without any concat)
        self.final =nn.Sequential(nn.Conv2d(filters[0], n_classes, 1))           #channel 64-21,kernel 1x1

        self.drop=nn.Dropout(p=0.25)


    def forward(self, inputs):
        # print("inputs: ",inputs.shape)
        conv1 = self.conv1(inputs)
        conv1_1=self.conv1_1(conv1)
        # print("conv1_1: ",conv1.shape)
        conv1_d = self.drop(conv1_1)
        inputs1=self.shortcut1(inputs)
        # print("inputs1.shape ",inputs1.shape)

        conv1 = conv1_d + inputs1
        # print("after addition conv1.shape: ",conv1.shape)

        conv2 = self.conv2(conv1)
        # print("conv2: ", conv2.shape)
        conv2_2=self.conv2_2(conv2)
        # print("conv2_2: ", conv2_2.shape)
        conv2_d = self.drop(conv2_2)
        inputs2=self.shortcut2(conv1)
        # print("inputs2.shape ", inputs2.shape)
        conv2 = conv2_d + inputs2
        # print("after addition conv2.shape",conv2.shape)

        conv3 = self.conv3(conv2)
        # print("conv2: ",conv2.shape)
        conv3_3 = self.conv3_3(conv3)
        conv3_d = self.drop(conv3_3)
        inputs3 = self.shortcut3(conv2)
        # print("inputs2.shape ", inputs2.shape)
        conv3 = conv3_d + inputs3
        # print("after addition conv3.shape", conv3.shape)


        center = self.center(conv3)
        center_1=self.center_1(center)
        # print("center.shape ", center.shape)
        center_d = self.drop(center_1)
        inputs_center = self.shortcut_center(conv3)
        # print("inputs_center.shape ", inputs_center.shape)
        center=center_d+inputs_center
        # print("after addition center.shape: ",center.shape)

        up3 = self.up_sampling3(center)
        # print("up3: ",up3.shape)
        concatenate3 = torch.cat([conv3, up3], 1)
        up3_conv=self.up_conv3(concatenate3)
        up3_conv2=self.up_conv3_3(up3_conv)
        up3_d=self.drop(up3_conv2)
        up3_inputs=self.up_shortcut3(concatenate3)
        up3=up3_inputs+up3_d

        up2 = self.up_sampling2(up3)
        # print("up2: ", up2.shape)
        concatenate2 = torch.cat([conv2, up2], 1)
        up2_conv = self.up_conv2(concatenate2)
        up2_conv2 = self.up_conv2_2(up2_conv)
        up2_d = self.drop(up2_conv2)
        up2_inputs = self.up_shortcut2(concatenate2)
        up2 = up2_inputs + up2_d


        up1 = self.up_sampling1(up2)
        # print("up1: ",up1.shape)
        concatenate1 = torch.cat([conv1, up1], 1)
        up1_conv = self.up_conv1(concatenate1)
        up1_conv2 = self.up_conv1_1(up1_conv)
        up1_d = self.drop(up1_conv2)
        up1_inputs = self.up_shortcut1(concatenate1)
        up1 = up1_inputs + up1_d

        final = self.final(up1)
        # print("final: ",final.shape)
        return final

import torch.nn as nn

from ptsemseg.models.utils import unetConv2, unetUp,unetfirstConv2


class unet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        # filters = [128,256,512,1024,2048]
        filters = [int(x / self.feature_scale) for x in filters]
        print(filters)
        #print("unet_initial_mark1")
        # downsampling

        #TODO: DOUBLE THE BASE FILTER NUMBER
        #TODO: RESIDUAL CONNECTION
        #TODO: MOVE DROPOUT LAYERS

        #TODO: TUNE DOWN THE LEARNING RATE WHEN TRAINING LOSS FLUCTUATES
        # print("unet_initial_mark1.2")
        self.conv1 = unetfirstConv2(self.in_channels, filters[0], self.is_batchnorm)  # channel 3-16
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.half1=nn.Conv2d(filters[0],filters[0], 2, 2, 0)
        # print("unet_initial_mark1.3")
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)  #channel 16-32
        # print("unet_initial_mark1.4")
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.half2 = nn.Conv2d(filters[1], filters[1], 2, 2, 0)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm) #channel 32-64
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.half3 = nn.Conv2d(filters[2], filters[2], 2, 2, 0)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)  #channel 64-128
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.half4 = nn.Conv2d(filters[3], filters[3], 2, 2, 0)



        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)  #channel 128-256

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)   #channel 256-128,problem
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)    #channel 128-64
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)   #channel 64-32
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)     #channel 32-16

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)             #channel 16-21,kernel 1x1
        #print("unet_initial_mark1.5")
        self.drop=nn.Dropout(p=0.3)
        self.shortcut1= nn.Sequential(nn.Conv2d(self.in_channels, filters[0], 1, 1, 0),nn.BatchNorm2d(filters[0]))
        self.shortcut2 = nn.Sequential(nn.Conv2d(filters[0], filters[1], 1, 2, 0),nn.BatchNorm2d(filters[1]))
        self.shortcut3 =nn.Sequential(nn.Conv2d(filters[1], filters[2], 1, 2, 0),nn.BatchNorm2d(filters[2]))
        self.shortcut4= nn.Sequential(nn.Conv2d(filters[2], filters[3], 1, 2, 0),nn.BatchNorm2d(filters[3]))
        self.shortcut_center=nn.Sequential(nn.Conv2d(filters[3], filters[4],1, 2, 0),nn.BatchNorm2d(filters[4]))
        # self.upcut4=nn.Conv2d(filters[4],filters[3],1,1,0)

    def forward(self, inputs):
        # print("inputs: ",inputs.shape)
        conv1 = self.conv1(inputs)
        # print("conv1: ",conv1.shape)
        #print("unet_initial_mark2")
        inputs1=self.shortcut1(inputs)
        # print("inputs1.shape ",inputs1.shape)
        # conv1=self.drop(conv1)
        conv1 = conv1 + inputs1
        # maxpool1 = conv1         #conv1-maxpool1

        # print("maxpool1.shape ",conv1.shape)

        #print("unet_initial_mark2.1")
        # print(maxpool1.shape)
        conv2 = self.conv2(conv1)            #maxpool1-conv2
        # print("conv2: ",conv2.shape)
        inputs2=self.shortcut2(conv1)
        # print("inputs2.shape ", inputs2.shape)
        # conv2=self.drop(conv2)
        conv2 = conv2 + inputs2
        # maxpool2 = self.half2(conv2)        #conv1-maxpool1
        # print("maxpool2.shape ",conv2.shape)


        conv3 = self.conv3(conv2)
        # print("conv3: ",conv3.shape)
        inputs3 = self.shortcut3(conv2)
        # print("inputs3.shape ", inputs3.shape)
        # conv3=self.drop(conv3)
        conv3=conv3+inputs3
        # maxpool3 = self.half3(conv3)
        # print("maxpool3.shape ", conv3.shape)

        conv4 = self.conv4(conv3)
        # print("conv4.shape: ",conv4.shape)
        inputs4 = self.shortcut4(conv3)
        # print("inputs4.shape ", inputs4.shape)
        conv4=self.drop(conv4)
        conv4 = conv4 + inputs4
        # maxpool4 = self.half4(conv4)
        # print("maxpool4.shape ", conv4.shape)

        #print("unet_initial_mark2.2")
        center = self.center(conv4)          #now channel is 256
        # print("center.shape ", center.shape)
        inputs_center=self.shortcut_center(conv4)
        # print("inputs_center.shape ", inputs_center.shape)
        center=self.drop(center)
        center=center+inputs_center
        # print("after addition center.shape: ",center.shape)

        #print("unet_initial_mark2.3")

        up4 = self.up_concat4(conv4, center)     #center 256-128 channels, concat with 128channels conv4
        # centers=self.upcut4(center)
        # up4=up4+centers

        # print("unet_initial_mark2.4")
        # print("up4: ",up4.shape)
        up3 = self.up_concat3(conv3, up4)  #problem!!!
        # print("up3: ",up3.shape)
        # print("unet_initial_mark2.5")
        up2 = self.up_concat2(conv2, up3)
        # print("up2: ", up2.shape)
        up1 = self.up_concat1(conv1, up2)
        # print("up1: ",up1.shape)
        # print("unet_initial_mark2.9")

        final = self.final(up1)
        # print("final: ",final.shape)
        return final

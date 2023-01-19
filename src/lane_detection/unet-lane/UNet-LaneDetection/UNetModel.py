import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import cv2
import numpy as np
import random
from UNet_Mask_Label_Gen import define_region_of_interest


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1, stride=1
        )

        # Encoder side
        # First layer two convolution layers
        self.conv_down1 = self.double_conv(
            5, 32, 3
        )  # Change dependng on if it's colour or grayscale
        # Second layer two convolution layers
        self.conv_down2 = self.double_conv(32, 64, 3)
        # Third layer two convolution layers
        self.conv_down3 = self.double_conv(64, 128, 3)
        # Fourth layer two convolution layers
        self.conv_down4 = self.double_conv(128, 256, 3)
        # Fifth layer two convlution layers
        self.conv_down5 = self.double_conv(256, 512, 3)
        # Sixth layer two convolution layers
        self.conv_down6 = self.double_conv(512, 1024, 3)

        # Decoder Side
        # First decoder layer two convolution layers
        self.upsample1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.conv_up1 = self.double_conv(1024, 512, 3)
        # Second decoder layer two convolution layers
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.conv_up2 = self.double_conv(512, 256, 3)
        # Third decoder layer two convolution layers
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.conv_up3 = self.double_conv(256, 128, 3)
        # Fourth decoder layer two convolution layers
        self.upsample4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.conv_up4 = self.double_conv(128, 64, 3)
        # Fifth decoder layer two convolution layers
        self.upsample5 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2
        )
        self.conv_up5 = self.double_conv(64, 32, 3)
        # Sixth decoder layer two convolution layers
        self.conv_up6 = self.double_conv(32, 3, 3)

    def forward(
        self, x
    ):  # Use F.interpolate on the final output to resize back to the original size
        # Encoder
        x1 = self.conv_down1(x)  # Output from first double conv
        x1_pool = self.max_pool(x1)
        x2 = self.conv_down2(x1_pool)  # Output from second double conv
        x2_pool = self.max_pool(x2)
        x3 = self.conv_down3(x2_pool)  # Output from third double conv
        x3_pool = self.max_pool(x3)
        x4 = self.conv_down4(x3_pool)  # Output from fourth double conv
        x4_pool = self.max_pool(x4)
        x5 = self.conv_down5(x4_pool)  # Output from fifth double conv
        x5_pool = self.max_pool(x5)
        x6 = self.conv_down6(x5_pool)  # Output from sixth double conv

        # print(x6.shape)
        # Decoder
        # Decoder first layer
        x7 = self.upsample1(x6)
        (_, _, H, W) = x7.shape
        x5_cropped = self.crop(x5, (H, W))
        x8 = self.conv_up1(torch.cat((x5_cropped, x7), dim=1))
        # Decoder second layer
        x9 = self.upsample2(x8)
        (_, _, H, W) = x9.shape
        x4_cropped = self.crop(x4, (H, W))
        x10 = self.conv_up2(torch.cat((x4_cropped, x9), dim=1))
        # Decoder third layer
        x11 = self.upsample3(x10)
        (_, _, H, W) = x11.shape
        x3_cropped = self.crop(x3, (H, W))
        x12 = self.conv_up3(torch.cat((x3_cropped, x11), dim=1))
        # Decoder fourth layer
        x13 = self.upsample4(x12)
        (_, _, H, W) = x13.shape
        x2_cropped = self.crop(x2, (H, W))
        x14 = self.conv_up4(torch.cat((x2_cropped, x13), dim=1))
        # Decoder fifth layer
        x15 = self.upsample5(x14)
        (_, _, H, W) = x15.shape
        x1_cropped = self.crop(x1, (H, W))
        x16 = self.conv_up5(torch.cat((x1_cropped, x15), dim=1))

        # x15 = self.conv_1x1(x14)
        x17 = self.conv_1x1(x16)

        # print(x17.shape)

        # x16 = F.interpolate(x15,(720,1280))
        x18 = F.interpolate(x17, (180, 330))
        # return x16
        # return torch.sigmoid(x18)
        return x18

    def double_conv(self, in_c, out_c, k_size=3):
        conv_double = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, 1, 1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, k_size, 1, 1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        return conv_double

    def crop(self, image, target_size):
        width = target_size[1]
        height = target_size[0]
        (_, _, height_img, width_img) = image.shape

        delta_width = torch.div(width_img - width, 2, rounding_mode="floor")
        delta_height = height_img - height

        if (width_img - 2 * delta_width) > width:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width: width_img - delta_width - 1
            ]
        elif (width_img - 2 * delta_width) < width:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width - 1: width_img - delta_width
            ]
        else:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width: width_img - delta_width
            ]

        return cropped_image


class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1, stride=1
        )

        # Encoder side
        # First layer two convolution layers
        self.conv_down1 = self.double_conv(1, 32, 3)
        # Second layer two convolution layers
        self.conv_down2 = self.double_conv(32, 64, 3)
        # Third layer two convolution layers
        self.conv_down3 = self.double_conv(64, 128, 3)

        # Decoder Side
        # First decoder layer two convolution layers
        self.upsample4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.conv_up4 = self.double_conv(128, 64, 3)
        # Secondd decoder layer two convolution layers
        self.upsample5 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2
        )
        self.conv_up5 = self.double_conv(64, 32, 3)

    def forward(
        self, x
    ):  # Use F.interpolate on the final output to resize back to the original size
        # Encoder
        x1 = self.conv_down1(x)  # Output from first double conv
        x1_pool = self.max_pool(x1)
        x2 = self.conv_down2(x1_pool)  # Output from second double conv
        x2_pool = self.max_pool(x2)
        x3 = self.conv_down3(x2_pool)  # Output from third double conv

        # Decoder
        # Decoder first layer
        x4 = self.upsample4(x3)
        (_, _, H, W) = x4.shape
        x2_cropped = self.crop(x2, (H, W))
        x5 = self.conv_up4(torch.cat((x2_cropped, x4), dim=1))
        # Decoder second layer
        x6 = self.upsample5(x5)
        (_, _, H, W) = x6.shape
        x1_cropped = self.crop(x1, (H, W))
        x7 = self.conv_up5(torch.cat((x1_cropped, x6), dim=1))

        x8 = self.conv_1x1(x7)
        x9 = F.interpolate(x8, (160, 240))

        return x9

    def double_conv(self, in_c, out_c, k_size=3):
        conv_double = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, 1, 1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, k_size, 1, 1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        return conv_double

    def crop(self, image, target_size):
        width = target_size[1]
        height = target_size[0]
        (_, _, height_img, width_img) = image.shape

        delta_width = (width_img - width) // 2
        delta_height = height_img - height

        if (width_img - 2 * delta_width) > width:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width: width_img - delta_width - 1
            ]
        elif (width_img - 2 * delta_width) < width:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width - 1: width_img - delta_width
            ]
        else:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width: width_img - delta_width
            ]

        return cropped_image


class UNet_Prob(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1, stride=1
        )

        # Encoder side
        # First layer two convolution layers
        self.conv_down1 = self.double_conv(
            5, 32, 3
        )  # Change dependng on if it's colour or grayscale
        # Second layer two convolution layers
        self.conv_down2 = self.double_conv(32, 64, 3)
        # Third layer two convolution layers
        self.conv_down3 = self.double_conv(64, 128, 3)
        # Fourth layer two convolution layers
        self.conv_down4 = self.double_conv(128, 256, 3)
        # Fifth layer two convlution layers
        self.conv_down5 = self.double_conv(256, 512, 3)
        # Sixth layer two convolution layers
        self.conv_down6 = self.double_conv(512, 1024, 3)

        # Decoder Side
        # First decoder layer two convolution layers
        self.upsample1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.upsample1_p = nn.ConvTranspose2d(
            in_channels=1024, out_channels=1024, kernel_size=2, stride=2
        )
        self.conv_up1 = self.double_conv(1024, 512, 3)
        # Second decoder layer two convolution layers
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.upsample2_p = nn.ConvTranspose2d(
            in_channels=512, out_channels=512, kernel_size=2, stride=2
        )
        self.conv_up2 = self.double_conv(512, 256, 3)
        # Third decoder layer two convolution layers
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.upsample3_p = nn.ConvTranspose2d(
            in_channels=256, out_channels=256, kernel_size=2, stride=2
        )
        self.conv_up3 = self.double_conv(256, 128, 3)
        # Fourth decoder layer two convolution layers
        self.upsample4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.upsample4_p = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=2, stride=2
        )
        self.conv_up4 = self.double_conv(128, 64, 3)
        # Fifth decoder layer two convolution layers
        self.upsample5 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=2, stride=2
        )
        self.upsample5_p = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=2
        )
        self.conv_up5 = self.double_conv(64, 32, 3)
        # Sixth decoder layer two convolution layers
        self.conv_up6 = self.double_conv(32, 3, 3)

    def forward(
        self, x, prob=0
    ):  # Use F.interpolate on the final output to resize back to the original size
        # Encoder
        x1 = self.conv_down1(x)  # Output from first double conv
        x1_pool = self.max_pool(x1)
        x2 = self.conv_down2(x1_pool)  # Output from second double conv
        x2_pool = self.max_pool(x2)
        x3 = self.conv_down3(x2_pool)  # Output from third double conv
        x3_pool = self.max_pool(x3)
        x4 = self.conv_down4(x3_pool)  # Output from fourth double conv
        x4_pool = self.max_pool(x4)
        x5 = self.conv_down5(x4_pool)  # Output from fifth double conv
        x5_pool = self.max_pool(x5)
        x6 = self.conv_down6(x5_pool)  # Output from sixth double conv

        # print(x6.shape)
        # Decoder
        # Decoder first layer
        if random.random() < p:
            x7 = self.upsample1(x6)
            (_, _, H, W) = x7.shape
            x5_cropped = self.crop(x5, (H, W))
            x8 = self.conv_up1(torch.cat((x5_cropped, x7), dim=1))
        else:
            x7 = self.upsample1_p(x6)
            x8 = self.conv_up1(x7)
        # Decoder second layer
        if random.random() < p:
            x9 = self.upsample2(x8)
            (_, _, H, W) = x9.shape
            x4_cropped = self.crop(x4, (H, W))
            x10 = self.conv_up2(torch.cat((x4_cropped, x9), dim=1))
        else:
            x9 = self.upsample2_p(x8)
            x10 = self.conv_up2(x9)
        # Decoder third layer
        if random.random() < p:
            x11 = self.upsample3(x10)
            (_, _, H, W) = x11.shape
            x3_cropped = self.crop(x3, (H, W))
            x12 = self.conv_up3(torch.cat((x3_cropped, x11), dim=1))
        else:
            x11 = self.upsample3_p(x10)
            x12 = self.conv_up3(x11)
        # Decoder fourth layer
        if random.random() < p:
            x13 = self.upsample4(x12)
            (_, _, H, W) = x13.shape
            x2_cropped = self.crop(x2, (H, W))
            x14 = self.conv_up4(torch.cat((x2_cropped, x13), dim=1))
        else:
            x13 = self.upsample4_p(x12)
            x14 = self.conv_up4(x13)
        # Decoder fifth layer
        if random.random() < p:
            x15 = self.upsample5(x14)
            (_, _, H, W) = x15.shape
            x1_cropped = self.crop(x1, (H, W))
            x16 = self.conv_up5(torch.cat((x1_cropped, x15), dim=1))
        else:
            x15 = self.upsample5_p(x14)
            x16 = self.conv_up5(x15)

        # x15 = self.conv_1x1(x14)
        x17 = self.conv_1x1(x16)

        # x16 = F.interpolate(x15,(720,1280))
        x18 = F.interpolate(x17, (180, 330))
        # return x16
        return x18

    def double_conv(self, in_c, out_c, k_size=3):
        conv_double = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, 1, 1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, k_size, 1, 1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        return conv_double

    def crop(self, image, target_size):
        width = target_size[1]
        height = target_size[0]
        (_, _, height_img, width_img) = image.shape

        delta_width = (width_img - width) // 2
        delta_height = height_img - height

        if (width_img - 2 * delta_width) > width:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width: width_img - delta_width - 1
            ]
        elif (width_img - 2 * delta_width) < width:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width - 1: width_img - delta_width
            ]
        else:
            cropped_image = image[
                :, :, delta_height:height_img, delta_width: width_img - delta_width
            ]

        return cropped_image


if __name__ == "__main__":
    weight_pth = "/Users/jasonyuan/Desktop/UNet Weights/unet_model_batch64_scheduled_lr0.05_epochs40_e14_best.pt"

    model = UNet()
    model.load_state_dict(torch.load(
        weight_pth, map_location=torch.device("cpu")))
    # model.to("cpu")
    model.eval()
    dummy_input = torch.ones(1, 5, 180, 330, device="cpu")

    out = model(dummy_input)
    # print(out.shape)
    # print(out)

    # torch.onnx.export(model,
    #                  dummy_input,
    #                  "/Users/jasonyuan/Desktop/unet_new.onnx",
    #                  opset_version=11,
    #                  do_constant_folding=True,
    #                  input_names=["Inputs"],
    #                  output_names=["Outputs"])
    #
    # print("Onnx conversion complete")

#     # test_tensor = torch.randn((5,3,160,240))
#     # unet = UNet2()
#     # out = unet(test_tensor)
#     # print(out.shape)
#
#     input_img = cv2.imread("/Users/jasonyuan/OneDrive/Seattle Lane Driving Data/Lane539.jpg")
#     roi2,M2,M_inv2 = define_region_of_interest(input_img)
#     # ground_truth = cv2.imread("/Users/jasonyuan/Desktop/UNet_Data/Dataset/labels/Lane_Label_1062.png",cv2.IMREAD_GRAYSCALE)
#     input_img_copy = cv2.resize(input_img,(330,180),interpolation=cv2.INTER_AREA)
#     roi,M,M_inv = define_region_of_interest(input_img_copy)
#     # print(roi.shape)
#     roi_copy = np.copy(roi)
#     roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
#     roi = roi.reshape(1,1,180,330)
#     roi = roi/255.
#     input = torch.Tensor(roi)
#     # input = torch.Tensor((cv2.cvtColor(input_img_copy,cv2.COLOR_BGR2GRAY)/255.).reshape(1,1,180,330))
#
#     unet = UNet()
#     unet.load_state_dict(torch.load("/Users/jasonyuan/Desktop/UNet_Data/UNet Weights/unet_model_batch16_lr0.001_epochs65_Best.pt",map_location=torch.device("cpu")))
#     unet.eval()
#     output = unet.forward(input)
#     # print(output)
#     # print(unet)
#
#     output = torch.sigmoid(output)
#     output = output.detach().numpy()
#     pred_mask = np.where(output>0.5,1,0)
#
#     # print(output)
#     # print(ground_truth.shape)
#     # print(pred_mask.size())
#     pred_mask = (pred_mask.squeeze(0)).transpose(1,2,0).squeeze().astype('float32')
#     pred_mask = cv2.resize(pred_mask,(1280,720),interpolation=cv2.INTER_AREA)
#
#     overlayed_mask = np.copy(roi2)
#     # overlayed_mask = np.copy(input_img)
#     overlayed_mask[np.where(pred_mask==1)[0],np.where(pred_mask==1)[1],2] = 255
#     overlayed_mask[np.where(pred_mask==1)[0],np.where(pred_mask==1)[1],1] = 0
#     overlayed_mask[np.where(pred_mask==1)[0],np.where(pred_mask==1)[1],0] = 0
#     overlayed_mask = cv2.warpPerspective(overlayed_mask,M_inv2,(1280,720),cv2.INTER_LINEAR)
#
#     overlayed = np.copy(input_img)
#     overlayed[np.where(overlayed_mask != [0,0,0])] = overlayed_mask[np.where(overlayed_mask != [0,0,0])]
#     overlayed = cv2.resize(overlayed,(1280,720),interpolation=cv2.INTER_AREA)
#
#     cv2.imshow("Input",input_img)
#     cv2.imshow("Input 2",input_img_copy)
#     # cv2.imshow("Ground Truth",ground_truth)
#     cv2.imshow("Predicted",(pred_mask*255).squeeze().astype("uint8"))
#     cv2.imshow("Overlayed",overlayed)
#     cv2.waitKey(0)

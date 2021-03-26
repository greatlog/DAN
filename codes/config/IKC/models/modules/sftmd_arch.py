"""
architecture for sftmd
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, in_nc=3, nf=64, code_len=10, use_bias=True):
        super(Predictor, self).__init__()

        self.ConvNet = nn.Sequential(
            *[
                nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    nf, code_len, kernel_size=5, stride=1, padding=2, bias=use_bias
                ),
                nn.LeakyReLU(0.2, True),
            ]
        )
        #   self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        return flat.view(flat.size()[:2])  # torch size: [B, code_len]


class Corrector(nn.Module):
    def __init__(self, in_nc=3, nf=64, code_len=10, use_bias=True):
        super(Corrector, self).__init__()

        self.ConvNet = nn.Sequential(
            *[
                nn.Conv2d(in_nc, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=use_bias),
                nn.LeakyReLU(0.2, True),
            ]
        )

        self.code_dense = nn.Sequential(
            *[
                nn.Linear(code_len, nf, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                nn.Linear(nf, nf, bias=use_bias),
                nn.LeakyReLU(0.2, True),
            ]
        )

        self.global_dense = nn.Sequential(
            *[
                nn.Conv2d(
                    nf * 2, nf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias
                ),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=use_bias
                ),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    nf, code_len, kernel_size=1, stride=1, padding=0, bias=use_bias
                ),
            ]
        )

        self.nf = nf
        self.globalPooling = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, input, code, res=False):
        conv_input = self.ConvNet(input)
        B, C_f, H_f, W_f = conv_input.size()  # LR_size

        conv_code = (
            self.code_dense(code)
            .view((B, self.nf, 1, 1))
            .expand((B, self.nf, H_f, W_f))
        )  # h_stretch
        conv_mid = torch.cat((conv_input, conv_code), dim=1)
        code_res = self.global_dense(conv_mid)

        # Delta_h_p
        flat = self.globalPooling(code_res)
        Delta_h_p = flat.view(flat.size()[:2])

        if res:
            return Delta_h_p
        else:
            return Delta_h_p + code


class SFT_Layer(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Layer, self).__init__()
        self.mul_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(para + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, para_maps):
        cat_input = torch.cat((feature_maps, para_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class SFT_Residual_Block(nn.Module):
    def __init__(self, nf=64, para=10):
        super(SFT_Residual_Block, self).__init__()
        self.sft1 = SFT_Layer(nf=nf, para=para)
        self.sft2 = SFT_Layer(nf=nf, para=para)
        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, feature_maps, para_maps):
        fea1 = F.relu(self.sft1(feature_maps, para_maps))
        fea2 = F.relu(self.sft2(self.conv1(fea1), para_maps))
        fea3 = self.conv2(fea2)
        return torch.add(feature_maps, fea3)


class SFTMD(nn.Module):
    def __init__(
        self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0
    ):
        super(SFTMD, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        self.conv1 = nn.Conv2d(in_nc, 64, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        for i in range(nb):
            self.add_module(
                "SFT-residual" + str(i + 1), SFT_Residual_Block(nf=nf, para=input_para)
            )

        self.sft = SFT_Layer(nf=64, para=input_para)
        self.conv_mid = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64 * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64 * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64 * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(
            in_channels=64,
            out_channels=out_nc,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=True,
        )

    def forward(self, input, ker_code):
        B, C, H, W = input.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand(
            (B_h, C_h, H, W)
        )  # kernel_map stretch

        fea_bef = self.conv3(
            self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(input))))
        )
        fea_in = fea_bef
        for i in range(self.num_blocks):
            fea_in = self.__getattr__("SFT-residual" + str(i + 1))(fea_in, ker_code_exp)
        fea_mid = fea_in
        # fea_in = self.sft_branch((fea_in, ker_code_exp))
        fea_add = torch.add(fea_mid, fea_bef)
        fea = self.upscale(self.conv_mid(self.sft(fea_add, ker_code_exp)))
        out = self.conv_output(fea)

        return torch.clamp(out, min=self.min, max=self.max)


class SFTMD_DEMO(nn.Module):
    def __init__(
        self, in_nc=3, out_nc=3, nf=64, nb=16, scale=4, input_para=10, min=0.0, max=1.0
    ):
        super(SFTMD_DEMO, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.reses = nb

        self.conv1 = nn.Conv2d(in_nc + input_para, 64, 3, stride=1, padding=1)
        self.relu_conv1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu_conv2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        for i in range(nb):
            self.add_module(
                "SFT-residual" + str(i + 1), SFT_Residual_Block(nf=64, para=input_para)
            )

        self.sft_mid = SFT_Layer(nf=nf, para=input_para)
        self.conv_mid = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.scale = scale
        if scale == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale == 3:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64 * 9,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_output = nn.Conv2d(
            in_channels=64,
            out_channels=out_nc,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )

    def forward(self, input, code, clip=False):
        B, C, H, W = input.size()
        B, C_l = code.size()
        code_exp = code.view((B, C_l, 1, 1)).expand((B, C_l, H, W))

        input_cat = torch.cat([input, code_exp], dim=1)
        before_res = self.conv3(
            self.relu_conv2(self.conv2(self.relu_conv1(self.conv1(input_cat))))
        )

        res = before_res
        for i in range(self.reses):
            res = self.__getattr__("SFT-residual" + str(i + 1))(res, code_exp)

        mid = self.sft_mid(res, code_exp)
        mid = F.relu(mid)
        mid = self.conv_mid(mid)

        befor_up = torch.add(before_res, mid)

        uped = self.upscale(befor_up)

        out = self.conv_output(uped)
        return torch.clamp(out, min=self.min, max=self.max) if clip else out

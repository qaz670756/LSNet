import torch.nn as nn
import torch
from models.CGNet import *
import warnings
from models.CoordAttention import CoordAtt
warnings.filterwarnings('ignore')

class up(nn.Module):
    def __init__(self, in_ch=3, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x

class light_siamese_backbone(nn.Module):
    def __init__(self, in_ch=None, num_blocks=None, cur_channels=None,
                 filters=None, dilations=None, reductions=None):
        super(light_siamese_backbone, self).__init__()
        norm_cfg = {'type': 'BN', 'eps': 0.001, 'requires_grad': True}
        act_cfg = {'type': 'PReLU', 'num_parameters': 32}
        self.inject_2x = InputInjection(1)  # down-sample for Input, factor=2
        self.inject_4x = InputInjection(2)  # down-sample for Input, factor=4
        # stage 0
        self.stem = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.stem.append(
                ContextGuidedBlock(
                    cur_channels[0], filters[0],
                    dilations[0], reductions[0],
                    skip_connect=(i != 0),
                    downsample=False,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)  # CG block
            )
            cur_channels[0] = filters[0]

        cur_channels[0] += in_ch
        self.norm_prelu_0 = nn.Sequential(
            build_norm_layer(cur_channels[0]),
            nn.PReLU(cur_channels[0]))

        # stage 1
        self.level1 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.level1.append(
                ContextGuidedBlock(
                    cur_channels[0] if i == 0 else filters[1],
                    filters[1], dilations[1], reductions[1],
                    downsample=(i == 0),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))  # CG block

        cur_channels[1] = 2 * filters[1] + in_ch
        self.norm_prelu_1 = nn.Sequential(
            build_norm_layer(cur_channels[1]),
            nn.PReLU(cur_channels[1]))

        # stage 2
        self.level2 = nn.ModuleList()
        for i in range(num_blocks[2]):
            self.level2.append(
                ContextGuidedBlock(
                    cur_channels[1] if i == 0 else filters[2],
                    filters[2], dilations[2], reductions[2],
                    downsample=(i == 0),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))  # CG block

        cur_channels[2] = 2 * filters[2]
        self.norm_prelu_2 = nn.Sequential(
            build_norm_layer(cur_channels[2]),
            nn.PReLU(cur_channels[2]))

        # stage 3
        self.level3 = nn.ModuleList()
        for i in range(num_blocks[3]):
            self.level3.append(
                ContextGuidedBlock(
                    cur_channels[2] if i == 0 else filters[3],
                    filters[3], dilations[3], reductions[3],
                    downsample=(i == 0),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))  # CG block

        cur_channels[3] = 2 * filters[3]
        self.norm_prelu_3 = nn.Sequential(
            build_norm_layer(cur_channels[3]),
            nn.PReLU(cur_channels[3]))

    def forward(self, x):
        # x = torch.cat([xA, xB], dim=0)
        # stage 0
        inp_2x = x  # self.inject_2x(x)
        inp_4x = self.inject_2x(x)
        for layer in self.stem:
            x = layer(x)
        x = self.norm_prelu_0(torch.cat([x, inp_2x], 1))
        x0_0A, x0_0B = x[:x.shape[0] // 2, :, :, :], x[x.shape[0] // 2:, :, :, :]

        # stage 1
        for i, layer in enumerate(self.level1):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.norm_prelu_1(torch.cat([x, down1, inp_4x], 1))
        x1_0A, x1_0B = x[:x.shape[0] // 2, :, :, :], x[x.shape[0] // 2:, :, :, :]

        # stage 2
        for i, layer in enumerate(self.level2):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.norm_prelu_2(torch.cat([x, down1], 1))
        x2_0A, x2_0B = x[:x.shape[0] // 2, :, :, :], x[x.shape[0] // 2:, :, :, :]

        # stage 3
        for i, layer in enumerate(self.level3):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.norm_prelu_3(torch.cat([x, down1], 1))
        x3_0A, x3_0B = x[:x.shape[0] // 2, :, :, :], x[x.shape[0] // 2:, :, :, :]

        return [x0_0A, x0_0B, x1_0A, x1_0B, x2_0A, x2_0B, x3_0A, x3_0B]

class diffFPN(nn.Module):
    def __init__(self, cur_channels=None, mid_ch=None,
                 dilations=None, reductions=None,
                 bilinear=True):
        super(diffFPN, self).__init__()
        # lateral convs for unifing channels
        self.lateral_convs = nn.ModuleList()
        for i in range(4):
            self.lateral_convs.append(
                cgblock(cur_channels[i] * 2, mid_ch * 2 ** i, dilations[i], reductions[i])
            )
        # top_down_convs
        self.top_down_convs = nn.ModuleList()
        for i in range(3, 0, -1):
            self.top_down_convs.append(
                cgblock(mid_ch * 2 ** i, mid_ch * 2 ** (i - 1), dilation=dilations[i], reduction=reductions[i])
            )

        # diff convs
        self.diff_convs = nn.ModuleList()
        for i in range(3):
            self.diff_convs.append(
                cgblock(mid_ch * (3 * 2 ** i), mid_ch * 2 ** i, dilations[i], reductions[i])
            )
        for i in range(2):
            self.diff_convs.append(
                cgblock(mid_ch * (3 * 2 ** i), mid_ch * 2 ** i, dilations[i], reductions[i])
            )
        self.diff_convs.append(
            cgblock(mid_ch * 3, mid_ch * 2,
                    dilation=dilations[0], reduction=reductions[0])
        )
        self.up2x = up(32, bilinear)

    def forward(self, output):
        tmp = [self.lateral_convs[i](torch.cat([output[i * 2], output[i * 2 + 1]], dim=1))
               for i in range(4)]

        # top_down_path
        for i in range(3, 0, -1):
            tmp[i - 1] += self.up2x(self.top_down_convs[3 - i](tmp[i]))

        # x0_1
        tmp = [self.diff_convs[i](torch.cat([tmp[i], self.up2x(tmp[i + 1])], dim=1)) for i in [0, 1, 2]]
        x0_1 = tmp[0]
        # x0_2
        tmp = [self.diff_convs[i](torch.cat([tmp[i - 3], self.up2x(tmp[i - 2])], dim=1)) for i in [3, 4]]
        x0_2 = tmp[0]
        # x0_3
        x0_3 = self.diff_convs[5](torch.cat([tmp[0], self.up2x(tmp[1])], dim=1))

        return x0_1, x0_2, x0_3

class diffFPN_thin(nn.Module):
    def __init__(self,cur_channels=None, mid_ch=None,
                 dilations=None, reductions=None,
                 bilinear=True):
        super(diffFPN_thin, self).__init__()
        # lateral convs for unifing channels
        self.lateral_convs = nn.ModuleList()
        for i in range(4):
            self.lateral_convs.append(
                cgblock(cur_channels[i] * 2, mid_ch * 2 ** i, dilations[i], reductions[i])
            )
        # top_down_convs
        self.top_down_convs = nn.ModuleList()
        for i in range(3, 0, -1):
            self.top_down_convs.append(
                cgblock(mid_ch * 2 ** i, mid_ch * 2 ** (i - 1), dilation=dilations[i], reduction=reductions[i])
            )

        # diff convs
        self.diff_convs = nn.ModuleList()
        for i in range(2):
            self.diff_convs.append(
                cgblock(mid_ch * (3 * 2 ** i), mid_ch * 2 ** i, dilations[i], reductions[i])
            )
        self.diff_convs.append(
            cgblock(mid_ch * 3, mid_ch * 2,
                    dilation=dilations[0], reduction=reductions[0])
        )
        self.up2x = up(32, bilinear)
    def forward(self,output):
        tmp = [self.lateral_convs[i](torch.cat([output[i * 2], output[i * 2 + 1]], dim=1))
               for i in range(4)]
        # top_down_path
        for i in range(3, 0, -1):
            tmp[i - 1] += self.up2x(self.top_down_convs[3 - i](tmp[i]))

        # x0_1
        tmp = [self.diff_convs[i](torch.cat([tmp[i], self.up2x(tmp[i + 1])], dim=1)) for i in [0, 1]]
        x0_1 = tmp[0]
        # x0_2
        x0_2 = self.diff_convs[2](torch.cat([tmp[0], self.up2x(tmp[1])], dim=1))

        return x0_1, x0_2

class denseFPN(nn.Module):
    def __init__(self, cur_channels=None, filters=None,
                 dilations=None, reductions=None, bilinear=True):
        super(denseFPN, self).__init__()
        self.conv0_1 = cgblock(cur_channels[0] * 2 + cur_channels[1] * 2, filters[0],
                               dilation=dilations[0], reduction=reductions[0])
        self.conv1_1 = cgblock(cur_channels[1] * 2 + cur_channels[2] * 2, filters[1],
                               dilation=dilations[1], reduction=reductions[1])
        self.conv2_1 = cgblock(cur_channels[2] * 2 + cur_channels[3] * 2, filters[2],
                               dilation=dilations[2], reduction=reductions[2])
        # self.conv3_1 = cgblock(cur_channels[3] * 2, cur_channels[2])
        self.Up1_1A = up(cur_channels[1], bilinear)
        self.Up1_1B = up(cur_channels[1], bilinear)
        self.Up2_1A = up(cur_channels[2], bilinear)
        self.Up2_1B = up(cur_channels[2], bilinear)
        self.Up3_1A = up(cur_channels[3], bilinear)
        self.Up3_1B = up(cur_channels[3], bilinear)

        self.conv0_2 = cgblock(cur_channels[0] * 2 + sum(filters[:2]), filters[0],
                               dilation=dilations[0], reduction=reductions[0])
        self.conv1_2 = cgblock(cur_channels[1] * 2 + sum(filters[1:3]), filters[1],
                               dilation=dilations[1], reduction=reductions[1])
        # self.conv2_2 = cgblock(filters[2] * 3 + filters[3], filters[2])
        self.Up1_2 = up(filters[1], bilinear)
        self.Up2_2 = up(filters[2], bilinear)

        self.conv0_3 = cgblock(cur_channels[0] * 2 + filters[0] * 2 + filters[1], filters[0],
                               dilation=dilations[0], reduction=reductions[0])
        # self.conv1_3 = cgblock(filters[1] * 4 + filters[2], filters[1])
        self.Up1_3 = up(filters[1], bilinear)
        self.Up2x = up(32, bilinear)
        # self.conv0_4 = cgblock(filters[0] * 5 + filters[1], filters[0])

    def forward(self, output):
        x0_0A, x0_0B, x1_0A, x1_0B, x2_0A, x2_0B, x3_0A, x3_0B = output

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_1A(x1_0A), self.Up1_1B(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_1A(x2_0A), self.Up2_1B(x2_0B)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_1A(x3_0A), self.Up3_1B(x3_0B)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_2(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_2(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_3(x1_2)], 1))

        return x0_1, x0_2, x0_3

class cam_head(nn.Module):
    def __init__(self, mid_ch=None, filters=None, out_ch=None, n=4):
        super(cam_head, self).__init__()
        assert filters is not None or mid_ch is not None
        ch = mid_ch if mid_ch is not None else filters[0]
        self.ca = CoordAtt(ch * n, ch * n, h=256, w=256, groups=16)
        self.conv_final = nn.Conv2d(ch * n, out_ch, kernel_size=1)

    def forward(self, x0_1, x0_2, x0_3=None):
        if x0_3 is not None:
            out = torch.cat([x0_1, x0_2, x0_3], 1)
        else:
            out = torch.cat([x0_1, x0_2], 1)
        out = self.ca(out)
        return self.conv_final(out)

class LSNet_diffFPN(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, mid_ch=32, out_ch=2, bilinear=True):
        super(LSNet_diffFPN, self).__init__()
        torch.nn.Module.dump_patches = True

        n1 = 32  # the initial number of channels of feature map
        filters = (n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16)
        num_blocks = (3, 3, 8, 12)
        dilations = (1, 2, 4, 8)
        reductions = (4, 8, 16, 32)
        cur_channels = [0, 0, 0, 0]
        cur_channels[0] = in_ch

        self.backbone = light_siamese_backbone(in_ch=in_ch, num_blocks=num_blocks,
                                               cur_channels=cur_channels,
                                               filters=filters, dilations=dilations,
                                               reductions=reductions)

        self.head = cam_head(mid_ch=mid_ch,out_ch=out_ch)

        self.FPN = diffFPN(cur_channels=cur_channels, mid_ch=mid_ch,
                           dilations=dilations, reductions=reductions, bilinear=bilinear)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, debug=False):

        output = self.backbone(x)

        x0_1, x0_2, x0_3 = self.FPN(output)

        out = self.head(x0_1, x0_2, x0_3)

        if debug:
            print_flops_params(self.backbone, [x], 'backbone')
            print_flops_params(self.FPN, [output], 'diffFPN')
            print_flops_params(self.head, [x0_1, x0_2, x0_3], 'head')

        return (x0_1, x0_2, x0_3, x0_3, out,)

class LSNet_thin_diffFPN(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, mid_ch=32, out_ch=2, bilinear=True):
        super(LSNet_thin_diffFPN, self).__init__()
        torch.nn.Module.dump_patches = True

        n1 = 32  # the initial number of channels of feature map
        filters = (n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16)
        num_blocks = (3, 3, 8, 12)
        dilations = (1, 2, 4, 8)
        reductions = (4, 8, 16, 32)
        cur_channels = [0, 0, 0, 0]
        cur_channels[0] = in_ch

        self.backbone = light_siamese_backbone(in_ch=in_ch, num_blocks=num_blocks,
                                               cur_channels=cur_channels,
                                               filters=filters, dilations=dilations,
                                               reductions=reductions)

        self.head = cam_head(filters=filters, out_ch=out_ch, n=3)

        self.FPN = diffFPN_thin(cur_channels=cur_channels, mid_ch=mid_ch,
                                dilations=dilations, reductions=reductions, bilinear=bilinear)

        # self.head = coordatt_head(mid_ch=mid_ch, out_ch=out_ch)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, debug=False):

        output = self.backbone(x)

        x0_1, x0_2 = self.FPN(output)

        out = self.head(x0_1, x0_2)

        if debug:
            print_flops_params(self.backbone, [x], 'backbone')
            print_flops_params(self.FPN, [output], 'diffFPN')
            print_flops_params(self.head, [x0_1, x0_2], 'head')

        return (x0_1, x0_2, x0_2, x0_2, out,)

class LSNet_denseFPN(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, mid_ch=32, out_ch=2, bilinear=True):
        super(LSNet_denseFPN, self).__init__()
        torch.nn.Module.dump_patches = True

        n1 = 32  # the initial number of channels of feature map
        filters = (n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16)
        num_blocks = (3, 3, 8, 12)
        dilations = (1, 2, 4, 8)
        reductions = (4, 8, 16, 32)
        cur_channels = [0, 0, 0, 0]
        cur_channels[0] = in_ch

        self.backbone = light_siamese_backbone(in_ch=in_ch, num_blocks=num_blocks,
                                               cur_channels=cur_channels,
                                               filters=filters, dilations=dilations,
                                               reductions=reductions)

        self.head = cam_head(mid_ch=mid_ch,out_ch=out_ch, n=3)

        self.FPN = denseFPN(cur_channels=cur_channels, filters = filters,
                           dilations=dilations, reductions=reductions, bilinear=bilinear)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, debug=False):

        output = self.backbone(x)

        x0_1, x0_2, x0_3 = self.FPN(output)

        out = self.head(x0_1, x0_2, x0_3)

        if debug:
            print_flops_params(self.backbone, [x], 'backbone')
            print_flops_params(self.FPN, [output], 'diffFPN')
            print_flops_params(self.head, [x0_1, x0_2, x0_3], 'head')

        return (x0_1, x0_2, x0_3, x0_3, out,)

def print_flops_params(model, input, name):
    flops, params = profile(model, inputs=input)
    print('-' * 10 + name.ljust(20, '-'))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


if __name__ == '__main__':
    from thop import profile

    model = LSNet_denseFPN()
    input = [torch.randn((2, 3, 256, 256))]
    print_flops_params(model, input, name='Total')
    model(input[0], debug=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from models.convNd import convNd

"""
VNet implementation adapted from github.com/mattmacy/vnet.pytorch
Updated to more closely follow the structure in the original UNet
allows outputs at pow(2) downsamples (for registration purposes)
"""


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan, inplace=True):
    if elu:
        return nn.ELU(inplace=inplace)
    else:
        return nn.PReLU(nchan)


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu, inchan=None):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        if inchan:
            self.conv1 = nn.Conv3d(inchan, nchan, kernel_size=5, padding=2)
        else:
            self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu, inchan=None):
    layers = []
    for i in range(depth):
        if i==0 and inchan:
            layers.append(LUConv(nchan, elu, inchan=inchan))
        else:
            layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu, use_4D=False, side_3D=False):
        super(InputTransition, self).__init__()
        self.outChans = outChans
        self.use_4D = use_4D
        self.side_3D = side_3D

        if use_4D:
            p = inChans // 2
            if side_3D:
                self.conv1 = convNd(in_channels=1, out_channels=outChans//2, num_dims=4, kernel_size=inChans, stride=(1, 1, 1, 1), padding=(0, p, p, p), use_bias=False)
                self.conv2 = nn.Conv3d(inChans, outChans//2, kernel_size=3, padding=1)
            else:
                self.conv1 = convNd(in_channels=1, out_channels=outChans, num_dims=4, kernel_size=inChans, stride=(1, 1, 1, 1), padding=(0, p, p, p), use_bias=False)
        else:
            self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        if self.side_3D:
            c2 = self.conv2(x)
        if self.use_4D:
            x = x[:,None,:]
        c1 = self.conv1(x)
        if self.use_4D:
            c1 = c1[:,:,0,:]
        if self.side_3D:
            c1 = torch.cat((c1,c2),1)
        norm1 = self.bn1(c1)
        #xn = torch.cat([x[:,0,:] for i in range(self.outChans)], 1) # treat as residual? check later whether that helps
        #out = self.relu1(torch.add(out, xn))
        out = self.relu1(norm1)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False, grow_fac=2):
        super(DownTransition, self).__init__()
        outChans = inChans*grow_fac
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class SegHead(nn.Module):
    def __init__(self, inChans, interm_ngf, classes, elu, dropout=False):
        super(SegHead, self).__init__()
        self.conv1 = LUConv(interm_ngf, elu, inchan=inChans)
        self.relu1 = ELUCons(elu, interm_ngf)
        self.bn1 = nn.BatchNorm3d(interm_ngf)
        self.segconv = nn.Conv3d(interm_ngf, classes, kernel_size=1, padding=0)
        self.segment_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        interm = self.relu1(self.bn1(self.conv1(x)))
        out = self.segment_softmax(self.segconv(interm))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False, joined_inchans=None):
        super(UpTransition, self).__init__()
        self.joined_inchans = joined_inchans
        self.outChans = outChans
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        if joined_inchans:
            #print(f'joined_inchans {joined_inchans}')
            self.relu2 = ELUCons(elu, joined_inchans, inplace=False)
        else:
            self.relu2 = ELUCons(elu, outChans, inplace=False)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu, inchan=joined_inchans)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        cout = self.up_conv(out)
        #print(f'debug_upt cout {cout.shape}')
        out = self.relu1(self.bn1(cout))
        #print(f'debug_upt out {out.shape}')
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        # only resconnect if dims match
        if self.joined_inchans == None or self.joined_inchans == self.outChans:
            out = self.relu2(torch.add(out, xcat))
        else:
            out = self.relu2(out)
        return out


class VNet_registration(nn.Module):
    """
    grid_spacing dictates the downsample level to output (i.e. 1=top, 2=2nd, 4=3rd, 8...)
    """
    def __init__(self, grid_spacing, inChans=2, outChans=3, ngf=64, elu=True, nll=False, use_4D=False, side_3D=False, use_dropout=True):
        super(VNet_registration, self).__init__()
        grid_spacing = grid_spacing[0]
        self.output_level = int(log2(grid_spacing))
        assert(grid_spacing < 32), "grid spacing > 16 not supported"
        assert(not (use_4D or side_3D)), "4D context and side_3D not supported in registration mode"

        self.in_tr = InputTransition(inChans, ngf, elu, use_4D=False, side_3D=False)
        self.in_tr2 = InputTransition(ngf, ngf, elu, use_4D=False, side_3D=False)
        self.down_tr32 = DownTransition(ngf, 2, elu, grow_fac=2)
        self.down_tr64 = DownTransition(ngf*2, 2, elu)
        self.down_tr128 = DownTransition(ngf*4, 2, elu, dropout=use_dropout)
        self.down_tr256 = DownTransition(ngf*8, 2, elu, dropout=use_dropout)
        self.up_tr256 = UpTransition(ngf*16, ngf*16, 2, elu, dropout=use_dropout)
        self.up_tr128 = UpTransition(ngf*16, ngf*8, 2, elu, dropout=use_dropout)
        self.up_tr64 = UpTransition(ngf*8, ngf*4, 2, elu)
        self.up_tr32 = UpTransition(ngf * 4, ngf * 2, 2, elu)

        reg_ngf = ngf * grid_spacing
        self.regconv1 = nn.Conv3d(reg_ngf*2, reg_ngf, kernel_size=3, padding=1)
        self.regelu = ELUCons(elu, reg_ngf)
        self.regconv2 = nn.Conv3d(reg_ngf, outChans, kernel_size=1)

    def forward(self, fixed, moving):
        x = torch.cat((fixed, moving), 1)
        out16 = self.in_tr2(self.in_tr(x))
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        if self.output_level < 4:
            out = self.up_tr256(out256, out128)
        if self.output_level < 3:
            out = self.up_tr128(out, out64)
        if self.output_level < 2:
            out = self.up_tr64(out, out32)
        if self.output_level < 1:
            out = self.up_tr32(out, out16)

        reg_1 = self.regconv1(out)
        reg_2 = self.regelu(reg_1)
        reg_out = self.regconv2(reg_2)

        return reg_out

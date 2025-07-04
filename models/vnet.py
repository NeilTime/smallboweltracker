import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convNd import convNd

"""
VNet implementation adapted from github.com/mattmacy/vnet.pytorch
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


class VNet(nn.Module):
    def __init__(self, inChans=3, dir_nc=500, seg_nc=2, ngf=64, first_ngf_fac=1, elu=True, nll=False, use_4D=False, side_3D=False, deep_supervision=False):
        super(VNet, self).__init__()
        self.first_ngf_fac = first_ngf_fac
        self.deep_supervision = deep_supervision
        self.in_tr = InputTransition(inChans, ngf * first_ngf_fac, elu, use_4D=use_4D, side_3D=side_3D)
        self.down_tr32 = DownTransition(ngf * first_ngf_fac, 1, elu, grow_fac=(2//first_ngf_fac))
        self.down_tr64 = DownTransition(ngf*2, 2, elu)
        self.down_tr128 = DownTransition(ngf*4, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(ngf*8, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(ngf*16, ngf*16, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(ngf*16, ngf*8, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(ngf*8, ngf*4, 1, elu)
        self.up_tr32 = UpTransition(ngf * 4, ngf * 2, 1, elu, joined_inchans=ngf * (1 + first_ngf_fac))

        self.segconv1 = nn.Conv3d(ngf*2, seg_nc, kernel_size=1, padding=0)
        #self.sigmoid = nn.Sigmoid()
        self.segment_softmax = nn.Softmax(dim=1)

        if deep_supervision:
            self.seghead_ds1 = SegHead(ngf * 4, ngf, seg_nc, elu)
            self.seghead_ds2 = SegHead(ngf * 8, ngf, seg_nc, elu)
            self.seghead_ds3 = SegHead(ngf * 16, ngf, seg_nc, elu)

        self.dirconv1 = nn.Conv3d(ngf*16, ngf*8, kernel_size=2, padding=0)
        self.direlu = ELUCons(elu, ngf*8)
        self.dirdo = nn.Dropout3d()
        self.dirmp = nn.MaxPool3d(2)
        self.dirconv2 = nn.Conv3d(ngf*8, dir_nc, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.n_shells = dir_nc // 500  # number of output shells, e.g. 1 for 500 shells

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out_l3 = self.up_tr256(out256, out128)
        out_l2 = self.up_tr128(out_l3, out64)
        out_l1 = self.up_tr64(out_l2, out32)
        out = self.up_tr32(out_l1, out16)
        seg1 = self.segconv1(out)
        seg_out = self.segment_softmax(seg1)
        #print(f'd_vnet: out.shape {out.shape}')
        #print(f'd_vnet: seg_out.shape {seg_out.shape}')
        if self.deep_supervision:
            seg_ds_1 = self.seghead_ds1(out_l1)
            seg_ds_2 = self.seghead_ds2(out_l2)
            seg_ds_3 = self.seghead_ds3(out_l3)

        if out256.shape[-1] == 4:  # if input was 64³ instead of 32³
            dirin = self.dirmp(out256)
        else:
            dirin = out256
        dir1 = self.dirconv1(dirin)
        dir2 = self.direlu(dir1)
        dir3 = self.dirdo(dir2)
        dir4 = self.dirconv2(dir3).flatten(2)         # [B, 500·S]
        B     = dir4.size(0)
        dir4  = dir4.view(B, self.n_shells, 500)      # [B,S,500]
        dir_out = self.softmax(dir4)                  # softmax over last dim

        if False:
            print(f'd_vnet: out256.shape {out256.shape}')
            print(f'd_vnet: dir1.shape {dir1.shape}')
            print(f'd_vnet: dir3.shape {dir3.shape}')
            print(f'd_vnet: dir4.shape {dir4.shape}')
            print(f'd_vnet: dir_out.shape {dir_out.shape}')

        if self.deep_supervision:
            return seg_out, dir_out, (seg_ds_1, seg_ds_2, seg_ds_3)
        else:
            return seg_out, dir_out

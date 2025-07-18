import torch
import torch.nn as nn
from models.convNd import convNd
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.vnet import VNet
from models.vnet_recentering import VNet_recentering
from .warm_restart_cosinescheduler import CosineAnnealingWarmUpRestarts


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'inverse_linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - (opt.niter_decay - epoch) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_step_size)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=1e-3, patience=opt.patience)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'warm_cosine':
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=opt.lr_decay_iters, T_up=opt.lr_decay_iters//15+1,
                                                  gamma=opt.lr_step_size, eta_max=opt.lr_max)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], n_dir_shells=1):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_notanh':
        net = ResnetGeneratorNoTanh(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'asym_resnet_9blocks_3crop':
        net = Asym3DResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, blocks_3d=3, n_blocks=9)
    elif netG == 'resnet3d_9blocks':
        net = ResnetGenerator3D(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_9blocks_notanh_zeropad':
        net = ResnetGeneratorNoTanhZeroPad(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_12blocks_notanh':
        net = ResnetGeneratorNoTanh(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=12)
    elif netG == 'resnet_18blocks_notanh':
        net = ResnetGeneratorNoTanh(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=18)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'dilated_19px':
        net = SimpleDilatedCNN(features=ngf, output_nc=output_nc, input_nc=input_nc)
    elif netG == 'dilated_33px':
        net = ExtendedDilatedCNN(features=ngf, output_nc=output_nc, input_nc=input_nc)
    elif netG == 'dilated_29px':
        net = DilatedCNNResTrunk(features=ngf, output_nc=output_nc, resblocks=5, input_nc=input_nc)
    elif netG == 'dilated_31px':
        net = DilatedCNNResTrunk(features=ngf, output_nc=output_nc, resblocks=3, input_nc=input_nc, cropfix=True)
    elif netG == 'vnet_4d':
        net = VNet(inChans=input_nc, seg_nc=output_nc, ngf=ngf, use_4D=True)
    elif netG[:15] == 'vnet34d_1stngf_':
        first_ngf_fac = int(netG[15:])
        net = VNet(inChans=input_nc, seg_nc=output_nc, ngf=ngf, first_ngf_fac=first_ngf_fac, use_4D=True, side_3D=True)
    elif netG[:15] == 'vnet_4d_1stngf_':
        first_ngf_fac = int(netG[15:])
        net = VNet(inChans=input_nc, seg_nc=output_nc, ngf=ngf, first_ngf_fac=first_ngf_fac, use_4D=True)
    elif netG[:18] == 'vnet34d_ds_1stngf_':
        first_ngf_fac = int(netG[18:])
        net = VNet(inChans=input_nc, seg_nc=output_nc, ngf=ngf, first_ngf_fac=first_ngf_fac, use_4D=True, side_3D=True, deep_supervision=True)
    elif netG[:18] == 'vnet_4d_ds_1stngf_':
        first_ngf_fac = int(netG[18:])
        net = VNet(inChans=input_nc, seg_nc=output_nc, ngf=ngf, first_ngf_fac=first_ngf_fac, use_4D=True, deep_supervision=True)
    elif netG == 'vnet_3d':
        net = VNet(inChans=input_nc, dir_nc=500 * n_dir_shells, seg_nc=output_nc, ngf=ngf, use_4D=False)
    elif netG == 'vnet_recentering':
        net = VNet_recentering(inChans=input_nc, seg_nc=output_nc, ngf=ngf, use_4D=False, use_dropout=use_dropout)
    elif netG == 'vnet_recentering_sepcoords':
        net = VNet_recentering(inChans=input_nc, seg_nc=output_nc, ngf=ngf, use_4D=False, use_dropout=use_dropout, sepcoords=True)
    elif netG[:15] == 'vnet_3d_1stngf_':
        first_ngf_fac = int(netG[15:])
        net = VNet(inChans=input_nc, seg_nc=output_nc, ngf=ngf, first_ngf_fac=first_ngf_fac, use_4D=False)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class SimpleDilatedCNN(nn.Module):
    """
    Adapted Jelmer's dilated CNN from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    output_nc controls the amount of shells being predicted (for multi-scale predictions)
    """
    def __init__(self, features=32, output_nc=1, sphere_samples=500, input_nc=1):
        super(SimpleDilatedCNN, self).__init__()
        C = features
        self.output_nc = 1#output_nc

        self.conv1 = nn.Conv3d(in_channels=input_nc, out_channels=C, kernel_size=3, dilation=1)  # 17
        self.bn1 = nn.BatchNorm3d(num_features=C)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1)  # 15
        self.bn2 = nn.BatchNorm3d(num_features=C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=2)
        self.bn3 = nn.BatchNorm3d(num_features=C)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=4)
        self.bn4 = nn.BatchNorm3d(num_features=C)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv3d(in_channels=C, out_channels=2*C, kernel_size=3, dilation=1)
        self.bn5 = nn.BatchNorm3d(num_features=2*C)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv3d(in_channels=2*C, out_channels=4*C, kernel_size=1, dilation=1)
        self.bn6 = nn.BatchNorm3d(num_features=4*C)
        self.relu6 = nn.ReLU()

        self.output_convs = []
        for output_layer in range(output_nc):
            self.output_convs.append(nn.Conv3d(in_channels=4*C, out_channels=sphere_samples, kernel_size=1))
        self.output_convs = nn.ModuleList(self.output_convs)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        h1 = self.relu1(self.bn1(self.conv1(input)))
        h2 = self.relu2(self.bn2(self.conv2(h1)))
        h3 = self.relu3(self.bn3(self.conv3(h2)))
        h4 = self.relu4(self.bn4(self.conv4(h3)))
        h5 = self.relu5(self.bn5(self.conv5(h4)))
        h6 = self.relu6(self.bn6(self.conv6(h5)))

        #print(f'h6 shape: {h6.shape}')
        output_shells = []
        for output_layer in range(self.output_nc):
            output_shells.append(self.output_convs[output_layer](h6))

        logits = torch.stack(output_shells, dim=1)
        #print(f'logits shape: {logits.shape}')
        out = self.softmax(logits)
        #print(f'out shape: {out.shape}')
        # temporary workaround to do inference in plus_segmentation
        segshape = [x for x in input.shape]
        segshape[1] = 2
        return torch.zeros((segshape)).cuda(), out


class ExtendedDilatedCNN(nn.Module):
    """
    Adapted Jelmer's dilated CNN from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    output_nc controls the amount of shells being predicted (for multi-scale predictions)
    extra convs before and after dilated section to pad receptive field to 33
    """
    def __init__(self, features=32, output_nc=1, sphere_samples=500, input_nc=1, preconvs=4, postconvs=3):
        super(ExtendedDilatedCNN, self).__init__()
        C = features
        self.output_nc = output_nc

        self.conv1 = nn.Conv3d(in_channels=input_nc, out_channels=C, kernel_size=3, dilation=1)  # 17
        self.bn1 = nn.BatchNorm3d(num_features=C)
        self.relu1 = nn.ReLU()

        self.preconvs = nn.ModuleList([])
        self.prebns = nn.ModuleList([])
        self.prerelus = nn.ModuleList([])
        for i in range(preconvs):
            self.preconvs.append(nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1))
            self.prebns.append(nn.BatchNorm3d(num_features=C))
            self.prerelus.append(nn.ReLU())
        self.postconvs = nn.ModuleList([])
        self.postbns = nn.ModuleList([])
        self.postrelus = nn.ModuleList([])
        for i in range(postconvs):
            self.postconvs.append(nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1))
            self.postbns.append(nn.BatchNorm3d(num_features=C))
            self.postrelus.append(nn.ReLU())

        self.conv2 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1)  # 15
        self.bn2 = nn.BatchNorm3d(num_features=C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=2)
        self.bn3 = nn.BatchNorm3d(num_features=C)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=4)
        self.bn4 = nn.BatchNorm3d(num_features=C)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv3d(in_channels=C, out_channels=2*C, kernel_size=3, dilation=1)
        self.bn5 = nn.BatchNorm3d(num_features=2*C)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv3d(in_channels=2*C, out_channels=4*C, kernel_size=1, dilation=1)
        self.bn6 = nn.BatchNorm3d(num_features=4*C)
        self.relu6 = nn.ReLU()

        self.output_convs = []
        for output_layer in range(output_nc):
            self.output_convs.append(nn.Conv3d(in_channels=4*C, out_channels=sphere_samples, kernel_size=1))
        self.output_convs = nn.ModuleList(self.output_convs)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        h1 = self.relu1(self.bn1(self.conv1(input)))
        hx = h1
        for i in range(len(self.preconvs)):
            hx = self.prerelus[i](self.prebns[i](self.preconvs[i](hx)))
        h2 = self.relu2(self.bn2(self.conv2(hx)))
        h3 = self.relu3(self.bn3(self.conv3(h2)))
        h4 = self.relu4(self.bn4(self.conv4(h3)))
        hy = h4
        for i in range(len(self.postconvs)):
            hy = self.postrelus[i](self.postbns[i](self.postconvs[i](hy)))
        h5 = self.relu5(self.bn5(self.conv5(hy)))
        h6 = self.relu6(self.bn6(self.conv6(h5)))

        output_shells = []
        for output_layer in range(self.output_nc):
            output_shells.append(self.output_convs[output_layer](h6))

        logits = torch.stack(output_shells, dim=1)
        out = self.softmax(logits)
        return out


class ExtendedDilatedCNN_4Dinconv(nn.Module):
    """
    Adapted Jelmer's dilated CNN from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    output_nc controls the amount of shells being predicted (for multi-scale predictions)
    extra convs before and after dilated section to pad receptive field to 33
    """
    def __init__(self, features=32, output_nc=1, sphere_samples=500, input_nc=1, preconvs=4, postconvs=3):
        super(ExtendedDilatedCNN_4Dinconv, self).__init__()
        C = features
        self.output_nc = output_nc

        assert input_nc > 2, "need at least 3 timeframes for 4D network"
        self.inconvs = nn.ModuleList([
            convNd(in_channels=1, out_channels=C, num_dims=4, kernel_size=3, stride=(1, 1, 1, 1), padding=0)
        ])
        self.inrelus = nn.ModuleList([nn.ReLU()])
        for i in range(input_nc//2 - 1):
            self.inconvs.append(
                convNd(in_channels=C, out_channels=C, num_dims=4, kernel_size=3, stride=(1, 1, 1, 1), padding=0))
            self.inrelus.append(nn.ReLU())

        self.bn1 = nn.BatchNorm3d(num_features=C)
        self.relu1 = nn.ReLU()

        self.preconvs = nn.ModuleList([])
        self.prebns = nn.ModuleList([])
        self.prerelus = nn.ModuleList([])
        for i in range(preconvs):
            self.preconvs.append(nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1))
            self.prebns.append(nn.BatchNorm3d(num_features=C))
            self.prerelus.append(nn.ReLU())
        self.postconvs = nn.ModuleList([])
        self.postbns = nn.ModuleList([])
        self.postrelus = nn.ModuleList([])
        for i in range(postconvs):
            self.postconvs.append(nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1))
            self.postbns.append(nn.BatchNorm3d(num_features=C))
            self.postrelus.append(nn.ReLU())

        self.conv2 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1)  # 15
        self.bn2 = nn.BatchNorm3d(num_features=C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=2)
        self.bn3 = nn.BatchNorm3d(num_features=C)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=4)
        self.bn4 = nn.BatchNorm3d(num_features=C)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv3d(in_channels=C, out_channels=2*C, kernel_size=3, dilation=1)
        self.bn5 = nn.BatchNorm3d(num_features=2*C)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv3d(in_channels=2*C, out_channels=4*C, kernel_size=1, dilation=1)
        self.bn6 = nn.BatchNorm3d(num_features=4*C)
        self.relu6 = nn.ReLU()

        self.output_convs = []
        for output_layer in range(output_nc):
            self.output_convs.append(nn.Conv3d(in_channels=4*C, out_channels=sphere_samples, kernel_size=1))
        self.output_convs = nn.ModuleList(self.output_convs)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        inx = input[:,None,:]
        print(f'debug: inxshape before inconvs: {inx.shape}')
        for i in range(len(self.inconvs)):
            inx = self.inrelus[i](self.preconvs[i](inx))

        print(f'debug: inxshape after inconvs: {inx.shape}')
        innorm = self.bn1(inx[:,0,:])
        print(f'debug: innorm shape: {innorm.shape}')

        h1 = self.relu1(self.bn1(self.conv1(innorm)))
        hx = h1
        for i in range(len(self.preconvs)):
            hx = self.prerelus[i](self.prebns[i](self.preconvs[i](hx)))
        h2 = self.relu2(self.bn2(self.conv2(hx)))
        h3 = self.relu3(self.bn3(self.conv3(h2)))
        h4 = self.relu4(self.bn4(self.conv4(h3)))
        hy = h4
        for i in range(len(self.postconvs)):
            hy = self.postrelus[i](self.postbns[i](self.postconvs[i](hy)))
        h5 = self.relu5(self.bn5(self.conv5(hy)))
        h6 = self.relu6(self.bn6(self.conv6(h5)))

        output_shells = []
        for output_layer in range(self.output_nc):
            output_shells.append(self.output_convs[output_layer](h6))

        logits = torch.stack(output_shells, dim=1)
        out = self.softmax(logits)
        return out


class DilatedCNNResTrunk(nn.Module):
    """
    Adaption of SimpleDilatedCNN with additional resblock before after dilated convs
    """
    def __init__(self, features=32, output_nc=1, sphere_samples=500, resblocks=5, input_nc=1, cropfix=False):
        super(DilatedCNNResTrunk, self).__init__()
        C = features
        self.output_nc = output_nc
        self.resblocks = resblocks
        self.cropfix = cropfix

        self.conv1 = nn.Conv3d(in_channels=input_nc, out_channels=C, kernel_size=3, dilation=1)  # 17
        self.bn1 = nn.BatchNorm3d(num_features=C)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=1)  # 15
        self.bn2 = nn.BatchNorm3d(num_features=C)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=2)
        self.bn3 = nn.BatchNorm3d(num_features=C)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, dilation=4)
        self.bn4 = nn.BatchNorm3d(num_features=C)
        self.relu4 = nn.ReLU()

        residual_trunk = []
        for i in range(resblocks):
            residual_trunk += [ResnetBlock3D(C, 'zero', norm_layer=nn.BatchNorm3d, use_dropout=False)]
        self.residual_trunk = nn.Sequential(*residual_trunk)

        self.conv5 = nn.Conv3d(in_channels=C, out_channels=2*C, kernel_size=3, dilation=1)
        self.bn5 = nn.BatchNorm3d(num_features=2*C)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv3d(in_channels=2*C, out_channels=4*C, kernel_size=1, dilation=1)
        self.bn6 = nn.BatchNorm3d(num_features=4*C)
        self.relu6 = nn.ReLU()

        self.output_convs = []
        for output_layer in range(output_nc):
            self.output_convs.append(nn.Conv3d(in_channels=4 * C, out_channels=sphere_samples, kernel_size=1))
        self.output_convs = nn.ModuleList(self.output_convs)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        h1 = self.relu1(self.bn1(self.conv1(input)))
        h2 = self.relu2(self.bn2(self.conv2(h1)))
        h3 = self.relu3(self.bn3(self.conv3(h2)))
        h4 = self.relu4(self.bn4(self.conv4(h3)))

        # apply the residual trunk
        h_restrunk = self.residual_trunk(h4)
        crop = self.resblocks
        if self.cropfix:
            crop = crop * 2  # resblock contains two convolutions. Bugged interface preserved to not break compatibility
        h_restrunk_crop = h_restrunk[:, :, crop:-crop, crop:-crop, crop:-crop]

        h5 = self.relu5(self.bn5(self.conv5(h_restrunk_crop)))
        h6 = self.relu6(self.bn6(self.conv6(h5)))

        output_shells = []
        for output_layer in range(self.output_nc):
            output_shells.append(self.output_convs[output_layer](h6))

        logits = torch.stack(output_shells, dim=1)
        out = self.softmax(logits)
        return out


class ResnetGeneratorNoTanhZeroPad(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='zero'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorNoTanhZeroPad, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetGeneratorNoTanh(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorNoTanh, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Asym3DResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    Input stage uses several 3d convolutions
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, blocks_3d=3, n_blocks=6, padding_type='zero'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            blocks_3d (int)     -- the number of 3d input stage convolutions
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(Asym3DResnetGenerator, self).__init__()
        self.blocks_3d = blocks_3d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_3d2d = [nn.Conv3d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                      nn.BatchNorm3d(ngf),
                      nn.ReLU(True)]

        for i in range(blocks_3d - 1):
            model_3d2d += [nn.Conv3d(ngf, ngf, kernel_size=3, padding=1, bias=use_bias),
                                nn.BatchNorm3d(ngf),
                                nn.ReLU(True)]

        model = [nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Sigmoid()]

        self.model_3d2d = nn.Sequential(*model_3d2d)
        self.model_2d = nn.Sequential(*model)

    def forward(self, input):
        features_2d = self.model_3d2d(input)
        result_list = []
        #print('features_2d.shape: {}'.format(features_2d.shape))
        for z in range(features_2d.shape[2]):
            feature_slice = features_2d[:, :, z, :, :]
            #print('sliceshape before model_2d: {}'.format(feature_slice.shape))
            feature_slice_result = self.model_2d(feature_slice)
            #print('sliceshape after model_2d: {}'.format(feature_slice_result.shape))
            result_list.append(feature_slice_result)
        result = torch.stack(result_list, dim=2)
        #print(result.shape)
        return result


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetBlock3D(nn.Module):
    """
    Modified resnetblock as in Identity Mappings in Deep Residual Networks (He et al. 2016): fully pre-activated
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        assert (padding_type == 'zero')
        p = 1

        conv_block += [norm_layer(dim, affine=True),
                       nn.ReLU(True),
                       nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(True),
                       nn.Conv3d(dim, dim, kernel_size=3, padding=p)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator3D(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=True, n_blocks=9, n_downsampling=2):
        assert (n_blocks >= 0)
        super(ResnetGenerator3D, self).__init__()
        self.ngf = ngf

        model = [nn.Conv3d(input_nc, self.ngf, kernel_size=3, padding=1),
                 norm_layer(self.ngf, affine=True),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv3d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(self.ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock3D(self.ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose3d(self.ngf * mult, int(self.ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(self.ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv3d(self.ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        res = self.model(input)
        return res


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

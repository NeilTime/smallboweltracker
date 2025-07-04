import torch
import itertools
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn.functional as F
# import fastgeodis as fg
from functools import lru_cache
from data.rt_colon_dataset import vox_coords_to_mm
import math
from guided_tracking_evaluator import compute_stochastic_trace
from torch.cuda.amp import autocast
import builtins
from contextlib import contextmanager

@contextmanager
def suppress_print():
    """Temporarily turn off the built-in print()."""
    original_print = builtins.print
    builtins.print = lambda *a, **k: None          # no-op
    try:
        yield
    finally:
        builtins.print = original_print            # restore

def _ensure_eval_defaults(opt):
    """Fill in the evaluator flags if they are missing (training run)."""
    defaults = dict(
        # -- tracker geometry -------------------------------------------------
        n_steps              = 500,
        step_size            = 0.5,        # mm
        n_candidates         = 1,
        start_seed           = '10.0,10.0,10.0',
        # -- stochastic tracing ----------------------------------------------
        stochastic_trace     = True,
        n_traces             = 16,
        test_time_augments   = 16,
        start_randomness_mm  = 1.0,
        confidence_thres     = 0.0,
        conformist_thres     = 15.0,        # vox
        moving_conf_average  = 1,
        rebuild_median       = False,
        doubleback_mindist   = 0.0,         # vox
        doubleback_slack_steps = 18,
        # -- sphere-sampling / shell merge -----------------------------------
        min_maxprob_dist     = 1.4142,      # ≈√2 ⇒ ~90° on the sphere
        scale_merge_metric   = 'mean',
        # -- out-of-volume handling ------------------------------------------
        hard_stop_oov        = False,
        disable_oov_slack    = False,
        # -- cohort mode (rare) ----------------------------------------------
        stepwise_cohort      = False,
        cohort_divergence_steps = 5,
        cohort_max_steps     = 30,
    )
    for k, v in defaults.items():
        if not hasattr(opt, k):
            setattr(opt, k, v)

class AbdominalTrackerPlusSegmentationModel(BaseModel):
    """
    TODO: implement this class.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        In this version, we only learn the forward function (A->B).
        A (source domain), B (target domain).
        Generators: G_A: A -> B
        Discriminators: -
        Forward loss: |G_A(A)) - A| (optional: squared)
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--loss_type', type=str, default='L2', help='Loss type (L1, L2, ce)')
            try:
                parser.add_argument('--backprop_crop', type=int, default=0, help='crop seg border by n pixels')
            except:
                pass

        return parser

    def __init__(self, opt):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.nshells = len(opt.gt_distances.split(',')) * (2 if opt.independent_dir else 1)
        # specify the training losses you want to print out. Training/test scripts call <BaseModel.get_current_losses>
        self.loss_names = [
            'G_A', 'MSE', 'BCE', 'val_top2_dist', 'GA_top2_dist', 'lr', 'L2reg', 
            'seg_BCE', 'seg_dice', 'seg_BCE_class', 'val_seg_BCE', 'val_seg_dice',
            'segs_ds', 'ds_tot',
            'selfconsistency', 'selfconsistency_raw', 'non_cl_selfconsistency_raw',
            'bidir_consistency_raw',     # mean mm distance between forward & reverse traces
            'bidir_consistency',          # raw • exp(-decay⋅ann_dev)
            'bidir',
        ]   
        # pre-define optional losses
        self.global_iter = 0
        self.loss_L2reg = self.opt.L2reg / 500 if self.opt.isTrain else 0
        self.loss_val_top2_dist = [np.nan for x in opt.gt_distances.split(',')]
        self.loss_val_msqd = 0
        self.loss_seg_BCE = 0
        self.loss_seg_dice = 0
        self.loss_val_seg_BCE = 0
        self.loss_val_seg_dice = 0
        self.loss_selfconsistency = 0
        self.loss_selfconsistency_raw = 0
        self.loss_non_cl_selfconsistency_raw = 0
        self.loss_bidir_consistency_raw = 0
        self.loss_bidir_consistency = 0
        self.loss_GA_top2_dist = [0, 0, 0]
        self.loss_segs_ds = torch.as_tensor([0, 0, 0])
        self.loss_ds_tot = 0
        self.loss_bidir = 0
        self.loss_seg_BCE_class = [0 for x in range(opt.output_nc)]
        # images to save/display. Training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            visual_names_A = ['viz_first_patch', 'viz_first_patch_gt', 'viz_first_patch_seg', 'viz_first_patch_seg_gt']
            visual_names_val = ['viz_valpatch', 'viz_valpatch_gt', 'viz_valpatch_seg', 'viz_valpatch_seg_gt'] #, 'viz_val_result', 'viz_valpatch_gt', 'viz_val_diff']

            self.visual_names = visual_names_A + visual_names_val  # combine visualizations for A and B
        else:
            self.visual_names = ['viz_first_patch']

        # models to save to the disk. Scripts call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A']
        else:
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_G(opt.input_nc,
                                        opt.output_nc,
                                        opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type,
                                        opt.init_gain, self.gpu_ids,
                                        n_dir_shells=self.nshells) # G_A(A) -> B

        if self.isTrain:
            # define loss functions
            self.MSELoss = torch.nn.MSELoss()
            self.BCELoss = torch.nn.BCELoss()
            if opt.loss_type == 'L1':
                self.criterionReg = torch.nn.L1Loss()
            elif opt.loss_type == 'L2':
                self.criterionReg = torch.nn.MSELoss()
            elif opt.loss_type.lower() == 'ce':
                self.criterionReg = torch.nn.BCELoss()
                self.mse_fac = 0
                self.abse_fac = 0
            elif 'cll' in opt.loss_type.lower():
                if len(opt.loss_type) == 3:
                    self.criterionReg = CenterlineLoss()
                else:
                    msefac = int(opt.loss_type[3:])
                    print(f'msefac: {msefac}')
                    self.criterionReg = CenterlineLoss(mse_fac=msefac)
            else:
                print('Unknown loss type [{}]!!!'.format(opt.loss_type))
                quit()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.optimizer.lower() == 'adam':
                self.optimizer_G = torch.optim.Adam(
                    itertools.chain(self.netG_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optimizer.lower() == 'sgd':
                self.optimizer_G = torch.optim.SGD(
                    itertools.chain(self.netG_A.parameters()), lr=opt.lr, momentum=opt.beta1)

            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_A = input['A'].to(self.device, non_blocking=True)

        # supply safe fall-backs when called in “eval” mode
        if 'B' in input:
            self.real_B = input['B'].to(self.device, non_blocking=True)
        else:
            # empty tensor so later losses see “no centre-line labels”
            self.real_B = torch.zeros(
                (self.real_A.size(0), self.opt.output_nc*3, 1),
                device=self.device
            )

        if 'C' in input:
            self.real_C = input['C'].to(self.device, non_blocking=True)
        else:
            # dummy segmentation map filled with “ignore” label −1
            self.real_C = torch.full(
                (self.real_A.size(0), 1,
                self.opt.patch_size, self.opt.patch_size, self.opt.patch_size),
                -1, device=self.device
            )

        if 'D' in input:
            self.clgt_available = input['D'].to(self.device, non_blocking=True)
        else:
            # flag “no centre-line voxels in this patch”
            self.clgt_available = torch.zeros(
                (self.real_A.size(0), 1, 1, 1, 1),
                dtype=torch.bool, device=self.device
            )

        if 'rotM' in input:
            self.rotM = input['rotM'].to(self.device, non_blocking=True)

        if 'center_mm' in input:
            # convert from mm to voxel coords
            self.ctr_mm = input['center_mm' ].to(self.device, non_blocking=True)
        
        if 'vol_id' in input:
            self.vol_id = input['vol_id'].to(self.device, non_blocking=True)

    def set_validation_input(self, batch_data, valvol):
        self.valvol = valvol
        self.valbatch, self.valbatch_gt, self.valbatch_seggt, self.vertices = batch_data
        self.valbatch = self.valbatch.cuda()
        self.valbatch_gt = self.valbatch_gt.cuda()
        self.valbatch_seggt = self.valbatch_seggt.cuda()

    def attach_dataset(self, dataset):
        """Give the model access to the louisDataset instance."""
        self.dataset = dataset

    def forward_field(self):
        """Run forward pass without cropping, generates vector field instead of vector"""
        self.fake_seg, self.fake_B = self.netG_A(self.real_A)  # G_A(A)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.deep_supervision:
            self.fake_C, self.fake_B, self.fake_DS_C = self.netG_A(self.real_A)  # G_A(A)
        else:
            self.fake_C, self.fake_B = self.netG_A(self.real_A)  # G_A(A)

        while self.fake_B.dim() > 3:
            self.fake_B = self.fake_B[..., 0]

        expected_shells = len(self.opt.gt_distances.split(',')) * (2 if self.opt.independent_dir else 1)

        assert self.fake_B.dim() == 3, f"fake_B should be 3-D, got {self.fake_B.dim()}-D"

        B, S, C = self.fake_B.shape
        assert C == 500, f"Direction histograms must have 500 bins, got {C}"
        assert S == expected_shells, f"Expected {expected_shells} shells, got {S}"

        self.fake_seg = [self.fake_C[:,c,:] for c in range(self.opt.output_nc)]
        if self.opt.deep_supervision:
            self.fake_DS_seg = [[x[:,c,:] for c in range(self.opt.output_nc)] for x in self.fake_DS_C]

        if self.isTrain and torch.sum(self.clgt_available) > 0:
            self.loss_GA_top2_dist = []
            for gt_shell in range(self.fake_B.shape[1]):
                topk = self.compute_top2_meandists(self.fake_B[self.clgt_available, gt_shell, :].detach(), self.real_B[self.clgt_available, gt_shell, :])
                self.loss_GA_top2_dist.append(topk)

    def backward_G(self):
        """Calculate the loss for generator G_A"""

        # loss [L[1||2]||ce](G_A(A), B)
        cl_patches_available = torch.sum(self.clgt_available) > 0
        fake_squeezed = self.fake_B[self.clgt_available.squeeze()]    # (Nc,S,500)
        real_squeezed = self.real_B[self.clgt_available.squeeze()]
        #print(f'debug_atpsm: fake_B shape {self.fake_B.shape}')
        #print(f'debug_atpsm: fake_squeezed shape {fake_squeezed.shape}')
        if cl_patches_available:
            losses = []
            for gt_shell in range(self.fake_B.shape[1]):
                fake_s = fake_squeezed[:, gt_shell, :]
                real_s = real_squeezed[:, gt_shell, :]

                # mask out placeholder (-1) bins once per shell
                valid = real_s >= 0
                if valid.any():
                    losses.append(self.BCELoss(fake_s[valid], real_s[valid]))

            if losses:
                self.loss_BCE_shell = torch.stack(losses)   # tensor [S]  (S = # shells)
                self.loss_BCE = self.loss_BCE_shell.mean()  # single scalar for optimiser
            else:
                self.loss_BCE_shell = torch.zeros(self.fake_B.size(1), device=self.device)
                self.loss_BCE = torch.tensor(0., device=self.device)

            self.loss_G_A = self.loss_BCE * self.opt.dir_bce_factor - self.opt.dir_bce_offset
            self.loss_MSE = self.MSELoss(fake_squeezed, real_squeezed)
            self.loss_BCE = self.BCELoss(fake_squeezed, real_squeezed)
            self.loss_L2reg = self.MSELoss(fake_squeezed, torch.zeros_like(fake_squeezed)) * self.opt.L2reg
        else:
            self.loss_G_A, self.loss_MSE, self.loss_BCE, self.loss_L2reg = 0, 0, 0, 0

        # self.loss_seg_BCE_class = [0 for x in range(self.opt.output_nc)]
        # # stack list of [B, D, H, W] predictions into [B, C, D, H, W]
        # all_preds = torch.stack(self.fake_seg, dim=1)  # [B, C, D, H, W]

        # # ground-truth: [B,1,D,H,W] -> [B,D,H,W]
        # gt = self.real_C.squeeze(1).long()
        # # one-hot:      [B,D,H,W] -> [B,D,H,W,C]
        # gt_onehot = F.one_hot(gt, num_classes=self.opt.output_nc)
        # # permute:      [B,D,H,W,C] -> [B,C,D,H,W]
        # gt_onehot = gt_onehot.permute(0, 4, 1, 2, 3).float()

        # # all_preds: [B, C, D, H, W]
        # # gt_onehot: [B, C, D, H, W]
        # # compute element‐wise BCE for every class & voxel
        # bce_map = F.binary_cross_entropy(all_preds, gt_onehot, reduction='none')  # [B, C, D, H, W]

        # # now average over batch+spatial dims to get one loss per class:
        # #   mean over dims 0,2,3,4 → shape [C]
        # loss_per_class = bce_map.mean(dim=[0,2,3,4]) * self.opt.seg_bce_factor

        # # store per‐class list
        # self.loss_seg_BCE_class = loss_per_class.tolist()

        # # global loss is the average of those C numbers
        # self.loss_seg_BCE = loss_per_class.mean()



        # segmentation loss component
        #print(f'fake_seg shape: {self.fake_seg.shape}')
        #print(f'real_seg shape: {self.real_C.shape}')
        self.loss_seg_BCE = 0
        self.loss_seg_BCE_class = [0 for x in range(self.opt.output_nc)]
        for c in range(self.opt.output_nc):
            fakeseg = self.fake_seg[c]
            realseg = ((self.real_C == c) * 1.0)
            realsegmask = (self.real_C == -1)
            
            if self.opt.backprop_crop > 0:
                bc = self.opt.backprop_crop
                # Only crop if the spatial dimensions are larger than 2*crop
                if fakeseg.shape[1] > 2*bc and realseg.shape[1] > 2*bc and realsegmask.shape[1] > 2*bc:
                    fakeseg = fakeseg[:, bc:-bc, bc:-bc, bc:-bc]
                    realseg = realseg[:, bc:-bc, bc:-bc, bc:-bc]
                    realsegmask = realsegmask[:, bc:-bc, bc:-bc, bc:-bc]
            
            fakeseg_squeezed = fakeseg.flatten()
            realseg_squeezed = realseg.flatten()
            realsegmask_squeezed = realsegmask.flatten()
            
            # Check if the mask and the predicted tensor have the same number of elements.
            if realsegmask_squeezed.numel() != fakeseg_squeezed.numel():
                print("Shape mismatch in segmentation loss for channel", c)
                print("fakeseg_squeezed.shape:", fakeseg_squeezed.shape)
                print("realsegmask_squeezed.shape:", realsegmask_squeezed.shape)
                # As a fallback, create a mask of the correct size filled with False.
                realsegmask_squeezed = torch.zeros_like(fakeseg_squeezed, dtype=torch.bool)
            
            # Set the predicted segmentation to 0 where the ground-truth mask indicates -1.
            fakeseg_squeezed[realsegmask_squeezed] = 0
            class_loss = self.BCELoss(fakeseg_squeezed, realseg_squeezed) * self.opt.seg_bce_factor
            self.loss_seg_BCE_class[c] = class_loss
            self.loss_seg_BCE += class_loss / self.opt.output_nc


        self.loss_segs_ds, self.loss_ds_tot = torch.as_tensor([0]), 0
        if self.opt.deep_supervision:
            self.loss_segs_ds, self.loss_ds_tot = self.calc_ds_loss(self.fake_DS_seg, self.real_C, (1/3,1/3,1/3))
            self.loss_segs_ds *= self.opt.seg_bce_factor
            self.loss_ds_tot *= self.opt.seg_bce_factor

        # # co-teaching supervision loss
        # selfconsistency, non_cl_selfconsistency = self.selfconsistency_loss()
        # self.loss_selfconsistency_raw, self.loss_non_cl_selfconsistency_raw = selfconsistency, non_cl_selfconsistency

        # if self.epoch > self.opt.selfconsistency_delay:
        #     self.loss_selfconsistency = selfconsistency * self.opt.selfconsistency_factor
        # else:
        #     self.loss_selfconsistency = 0
        
        # # bidirectional consistency loss
        # if self.epoch >= self.opt.bidir_delay:
        #     self.loss_bidir_consistency_raw, self.loss_bidir_consistency = self.compute_bidir_consistency()

        # if self.global_iter % self.opt.bidir_every == 0 and self.epoch >= self.opt.bidir_delay:
        #     # 1. direction predicted **in patch coords**
        #     assert self.fake_B.dim() == 3,  "fake_B should be (B,S,C)"
        #     assert self.fake_B.shape[-1] == len(self.vertices), "last dim must equal Nclasses"
        #     v_local = self.dir_from_spherical(self.fake_B[:, 0, :])         # (B,3)

        #     # 2. turn it into a world-space vector
        #     v_world = self.dir_from_spherical(self.fake_B[:, 0, :], frame='world')
        #     assert v_world.shape[-1] == 3

        #     # 3. walk Δ mm forward
        #     step_mm = self.opt.bidir_step_mm
        #     ctr_mm = self.ctr_mm.squeeze(-1)
        #     new_mm = self.ctr_mm + v_world * step_mm                 # (B,3)

        #     # 4. resample the *same-orientation* patch
        #     patch2 = []
        #     for vol_id, c_mm, R in zip(self.vol_id, new_mm, self.rotM):
        #         vol = self.dataset.volumes[int(vol_id)]
        #         c_mm_np = c_mm.detach().cpu().numpy() 
        #         im_patch, *_ = vol.get_patch(
        #             c_mm_np,
        #             rot_matrix_override=R.detach().cpu().numpy(),
        #             rotate=False          # orientation already fixed
        #         )
        #         patch2.append(im_patch)
        #     patch2 = torch.stack(patch2).to(self.device)

        #     # 5. second forward pass
        #     _, fake_B2 = self.netG_A(patch2)

        #     # drop the spatial dimensions exactly like in the forward pass
        #     fake_B2 = fake_B2[:, :, :, 0, 0, 0]              # → (B, S, C=500)
        #     v2_local = self.dir_from_spherical(fake_B2[:, 0, :]) 

        #     # 6. bidirectional losses
        #     loss_dir = 1 - F.cosine_similarity(v2_local, -v_local, dim=1).mean()
        #     loss_cyc = F.l1_loss(new_mm, self.ctr_mm + v_world * step_mm)  # trivial, optional

        #     self.loss_bidir = (loss_dir * self.opt.lambda_dir +
        #                     loss_cyc * self.opt.lambda_cycle)
        # else:
        #     self.loss_bidir = torch.tensor(0.0, device=self.device)

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_L2reg + self.loss_seg_BCE + self.loss_selfconsistency + self.loss_ds_tot # + self.loss_bidir_consistency + self.loss_bidir
        self.loss_G.backward()

    @torch.no_grad()
    def compute_seg_dice(self):
        """
        Compute the mean Dice coefficient for class-1 over the batch.

        Assumes:
        - self.fake_C: tensor of shape [B, C, D, H, W] (network’s softmax probabilities)
        - self.real_C: tensor of shape [B, D, H, W] with integer labels in {0,...,C-1}
        """
        eps = 1e-6

        # Predicted label map: take highest-score class at each voxel
        # result has shape [B, D, H, W]
        pred_labels = self.fake_C.argmax(dim=1)

        # Flatten spatial dims to [B, V]
        B = pred_labels.shape[0]
        pred_flat = pred_labels.view(B, -1).float()  # [B, V]
        gt_flat = self.real_C.view(B, -1).float()  # [B, V]

        # Build binary masks for class-1
        pred_mask = (pred_flat == 1).float()  # [B, V]
        gt_mask = (gt_flat == 1).float()  # [B, V]

        # Compute intersection and sums per sample
        intersection = (pred_mask * gt_mask).sum(dim=1)  # [B]
        pred_sum = pred_mask.sum(dim=1)  # [B]
        gt_sum = gt_mask.sum(dim=1)  # [B]

        # Dice per sample, then average
        dice_per_sample = (2*intersection + eps) / (pred_sum + gt_sum + eps)  # [B]
        return dice_per_sample.mean()  # scalar in [0,1]

    @torch.no_grad()
    def compute_bidir_consistency(self):
        """
        1. Use the full validation volume (self.valvol) to build a forward
        and reverse stochastic trace (code reused from evaluator).
        2. Return:
        - raw  : mean mm distance between the two traces where they overlap
        - scaled: raw * exp(-decay · annotation_error) * factor
        If self.valvol is absent (e.g. during the very first iterations), returns zeros.
        """
        if not hasattr(self, 'valvol'):
            return torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

        import guided_tracking_evaluator as gte
        gte.opt = self.valvol.opt            # give the evaluator its global “opt”

        # pick a single annotated centre-line key (0 is fine)
        _ensure_eval_defaults(self.valvol.opt)
        key = next(iter(self.valvol.sint_segs_dense_vox.keys()))
        gt_vox = self.valvol.sint_segs_dense_vox[key]
        spacing = self.valvol.spacing
        gt_mm = vox_coords_to_mm(gt_vox, spacing)

        # build a forward & reverse trace with existing evaluator code
        # forward + reverse median trace in one call
        with suppress_print():                             # ← silences the tracer
            median_trace_mm, *_ = gte.compute_stochastic_trace(
                self, self.valvol, key, gtdist_thres=False
            )

        # split into forward / reverse halves
        mid = len(median_trace_mm) // 2
        fw  = np.asarray(median_trace_mm[:mid+1])         # list → ndarray
        bw  = np.asarray(median_trace_mm[mid:])           # already reversed by evaluator
        L   = min(len(fw), len(bw))
        if L < 2:                                         # too short → skip
            return torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

        fw_t = torch.as_tensor(fw[:L], device=self.device, dtype=torch.float32)
        bw_t = torch.as_tensor(bw[:L], device=self.device, dtype=torch.float32)

        # raw distance (mean point-wise Euclidean)
        raw_dist = torch.norm(fw_t - bw_t, dim=1).mean()     # scalar [mm]

        # annotation deviation (for decay)
        gt_t  = torch.as_tensor(gt_mm[:L], device=self.device, dtype=torch.float32)
        ann_err = torch.norm(fw_t - gt_t, dim=1).mean().detach()  # scalar

        # scaled loss
        weight = math.exp(- self.opt.bidir_consistency_decay * ann_err.item())
        scaled = raw_dist * weight * self.opt.bidir_consistency_factor

        return raw_dist, scaled

    def selfconsistency_loss(self):
        """
        cast rays from center, check if high probability directions are foreground in the segmentation
        """
        assert self.opt.output_nc > 1, "selfconsistency assumes small intestine in class 1"
        assert self.opt.selfconsistency_range > 0, "selfconsistency range has to be positive nonzero"

        try:
            rays = self.fake_seg[1][:, self.coord_cache[:,:,0], self.coord_cache[:,:,1], self.coord_cache[:,:,2]]
        except AttributeError:
            self.coord_cache = torch.zeros((500, self.opt.selfconsistency_range, 3), dtype=torch.long)
            segshape = self.fake_seg[0][0, :].shape
            segcenter = (torch.as_tensor(segshape)-1) / 2.
            for d in range(self.opt.selfconsistency_range):
                self.coord_cache[:, d, :] = torch.round(segcenter + d * torch.as_tensor(self.vertices)).long()

            rays = self.fake_seg[1][:, self.coord_cache[:,:,0], self.coord_cache[:,:,1], self.coord_cache[:,:,2]]

        ray_dist_from_foreground = 1-rays
        if self.opt.raydist_zerofilter:
            rays_in_input = self.real_A[:, 0, self.coord_cache[:, :, 0], self.coord_cache[:, :, 1], self.coord_cache[:, :, 2]]
            zerocoords = rays_in_input == 0
            ray_dist_from_foreground[zerocoords] = 0
        raydist_integral = ray_dist_from_foreground.sum(-1) / self.opt.selfconsistency_range
        # clip to above-average rays
        ray_weights = (self.fake_B[:,0,:]-torch.mean(self.fake_B[:,0,:])).clamp(0,1)

        weighted_rays = ray_weights[self.clgt_available, :] * raydist_integral[self.clgt_available, :]
        raysum = torch.mean(weighted_rays) if torch.sum(self.clgt_available) > 0 else 0
        weighted_non_cl_rays = ray_weights[~self.clgt_available, :] * raydist_integral[~self.clgt_available, :]
        non_cl_raysum = torch.mean(weighted_non_cl_rays) if torch.sum(~self.clgt_available) > 0 else 0

        return raysum, non_cl_raysum

    def calc_ds_loss(self, fake_ds_segs, real_seg, weights=None):
        """
        Calculate segmentation loss for deep supervision heads
        """
        if weights is not None:
            assert len(weights) == len(fake_ds_segs), "fake_ds_segs length does not match weighting array lenght!"
        else:
            weights = [1 for x in range(len(fake_ds_segs))]

        seg_criterion = 'bce'  #TODO make dice an option
        if seg_criterion == 'dice':
            import torchgeometry
            crit = torchgeometry.losses.DiceLoss()
            pool = torch.nn.MaxPool3d(2, stride=2)
        else:
            crit = torch.nn.BCELoss()
            pool = torch.nn.AvgPool3d(2, stride=2)

        losses = [0 for x in range(len(weights))]
        loss_total = 0
        for c in range(self.opt.output_nc):
            real_seg_c = (real_seg == c) * 1.0
            gtcs = [pool(real_seg_c)]
            for i in range(len(fake_ds_segs)):
                loss = weights[i] * crit(fake_ds_segs[i][c].flatten(), gtcs[i].flatten()) / self.opt.output_nc
                loss_total += loss
                losses[i] += loss
                gtcs.append(pool(gtcs[i]))

        return losses, loss_total

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A the only thing to update
        self.optimizer_G.zero_grad()  # set G_A gradients to zero
        self.backward_G()  # calculate gradients for G_A
        self.optimizer_G.step()  # update G_A's weights
        self.loss_seg_dice = self.compute_seg_dice()
        self.global_iter += 1

    def compute_visuals(self):
        """Create validation image"""
        prevphase = self.opt.phase
        self.opt.phase = 'val'
        spacing = self.opt.isosample_spacing

        with torch.no_grad():
            if self.isTrain:
                valpatch_cpu = self.valbatch.cpu().numpy()[0, 0, :, :, :]
                firstpatch_cpu = self.real_A[0, 0, :, :, :].cpu().numpy()
                valpatch_seggt_cpu = (self.valbatch_seggt[0, :].cpu().numpy() == 1) * 1.0
                valpatch_seggt_cpu = np.squeeze(valpatch_seggt_cpu, axis=0)
                firstpatch_seggt_cpu = (self.real_C[0, :, :, :].cpu().numpy() == 1) * 1.0
                firstpatch_seggt_cpu = np.squeeze(firstpatch_seggt_cpu, axis=0)
                firstpatch_seg_cpu = self.fake_seg[1][0, :, :, :].cpu().numpy()
                if self.opt.output_nc == 4:
                    valpatch_seggt_cpu_2 = (self.valbatch_seggt[0, :].cpu().numpy() == 2) * 1.0
                    valpatch_seggt_cpu_3 = (self.valbatch_seggt[0, :].cpu().numpy() == 3) * 1.0
                    firstpatch_seggt_cpu_2 = (self.real_C[0, :, :, :].cpu().numpy() == 2) * 1.0
                    firstpatch_seggt_cpu_3 = (self.real_C[0, :, :, :].cpu().numpy() == 3) * 1.0
                    firstpatch_seg_cpu_2 = self.fake_seg[2][0, :, :, :].cpu().numpy()
                    firstpatch_seg_cpu_3 = self.fake_seg[3][0, :, :, :].cpu().numpy()


                self.viz_valpatch = self.triplanar_canvas(valpatch_cpu)
                self.viz_valpatch_gt = self.triplanar_canvas(valpatch_cpu)
                if self.opt.output_nc == 4:
                    self.viz_valpatch_seg_gt = self.triplanar_canvas(valpatch_seggt_cpu*4, valpatch_seggt_cpu_2*4, valpatch_seggt_cpu_3*4)
                    self.viz_first_patch_seg_gt = self.triplanar_canvas(firstpatch_seggt_cpu * 4, firstpatch_seggt_cpu_2 * 4, firstpatch_seggt_cpu_3 * 4)
                    self.viz_first_patch_seg = self.triplanar_canvas(firstpatch_seg_cpu * 4, firstpatch_seg_cpu_2 * 4, firstpatch_seg_cpu_3 * 4)
                else:
                    self.viz_valpatch_seg_gt = self.triplanar_canvas(valpatch_seggt_cpu * 4)
                    self.viz_first_patch_seg_gt = self.triplanar_canvas(firstpatch_seggt_cpu * 4)
                    self.viz_first_patch_seg = self.triplanar_canvas(firstpatch_seg_cpu * 4)

                self.viz_first_patch = self.triplanar_canvas(firstpatch_cpu)
                self.viz_first_patch_gt = self.triplanar_canvas(firstpatch_cpu)

                self.netG_A.eval()
                if self.opt.deep_supervision:
                    valseg, all_val_results, _ = self.netG_A(self.valbatch)  #TODO plot ds as well?
                else:
                    valseg, all_val_results = self.netG_A(self.valbatch)
                self.netG_A.train()

                valpatch_seg_cpu = valseg[0, 1, :].cpu().numpy()
                if self.opt.output_nc == 4:
                    valpatch_seg_cpu_2 = valseg[0, 2, :].cpu().numpy()
                    valpatch_seg_cpu_3 = valseg[0, 3, :].cpu().numpy()
                    self.viz_valpatch_seg = self.triplanar_canvas(valpatch_seg_cpu * 4, valpatch_seg_cpu_2 * 4, valpatch_seg_cpu_3 * 4)
                else:
                    self.viz_valpatch_seg = self.triplanar_canvas(valpatch_seg_cpu * 4)

                self.loss_val_seg_BCE = self.BCELoss(valseg[:, 1, :].flatten(), ((self.valbatch_seggt[:, :] == 1)*1.0).flatten())
                if not self.opt.binarize_segmentation:
                    print('note: val_seg_BCE loss only computed on small intestine')

                eps = 1e-6
                B = valseg.size(0)

                # predicted labels per voxel → shape [B, V]
                pred_labels = valseg.argmax(dim=1)

                # flatten predictions → [B, V]
                pred_flat = pred_labels.view(B, -1).float()

                # flatten ground truth labels → [B, V]
                # assumes self.valbatch_seggt is a label‐map of shape [B, D, H, W]
                gt_flat = self.valbatch_seggt.view(B, -1).float()

                # binary masks for class-1
                p1 = (pred_flat == 1).float()
                g1 = (gt_flat   == 1).float()

                # per-sample intersection and sums
                inter = (p1 * g1).sum(dim=1)   # [B]
                p_sum = p1.sum(dim=1)          # [B]
                g_sum = g1.sum(dim=1)          # [B]

                # Dice per sample, then mean
                dice_per = (2 * inter + eps) / (p_sum + g_sum + eps)  # [B]
                self.loss_val_seg_dice = dice_per.mean()  # scalar in [0,1]

                self.loss_val_top2_dist = []
                for gt_shell in range(self.fake_B.shape[1]):
                    #print(f'shape fake_B: {self.fake_B.shape}')
                    topk = self.compute_top2_meandists(all_val_results[:, gt_shell, :].squeeze(),
                                                       self.valbatch_gt[:, gt_shell, :])
                    self.loss_val_top2_dist.append(topk)

                while all_val_results.dim() > 3:
                    all_val_results = all_val_results[..., 0]

                val_results = all_val_results[0]               # shape (n_shells, 500)

                assert val_results.ndim in (1, 2),          \
                    f"val_results should be 1-D or 2-D, got shape {val_results.shape}"
                if val_results.ndim == 2:
                    assert val_results.shape[1] == 500,     \
                        f"Each histogram must have 500 bins, got {val_results.shape[1]}"
                else:  # single shell
                    assert val_results.shape[0] == 500,     \
                        f"Single histogram must have 500 bins, got {val_results.shape[0]}"

                # helper: turn 1-D tensor/list of bin IDs into (N,3) numpy array of xyz
                def _ids_to_vec(id_tensor):
                    if torch.is_tensor(id_tensor):
                        ids = id_tensor.detach().cpu().numpy().astype(int)
                    else:
                        ids = np.asarray(id_tensor, dtype=int)
                    return self.vertices[ids]          # vertices is (500,3) numpy array

                gt_dists = [int(x) for x in self.opt.gt_distances.split(',')]
                gt_dists = [int(np.round(0.5+x/spacing)) for x in gt_dists]
                firstpatch_is_on_cl = self.clgt_available[0]
                for gt_shell in range(self.fake_B.shape[1]):
                    val_result = val_results[gt_shell, :]
                    val_gt = self.valbatch_gt[0, gt_shell, :].cpu().numpy()
                    val_gt_loc = np.where(val_gt > val_gt.max()*0.5)[0]   # ← 1-D array
                    if firstpatch_is_on_cl:
                        firstpatch_gt_cpu = self.real_B[0, gt_shell, :].cpu().numpy()
                        fp_gt_loc = np.where(firstpatch_gt_cpu > firstpatch_gt_cpu.max()*0.5)[0]  # ← 1-D array

                    valgt_prob = val_gt[val_gt_loc]
                    val_gt_coords = self.get_top2_from_topk(torch.as_tensor(valgt_prob), torch.as_tensor(val_gt_loc))
                    topk_valres_prob, topk_valres_ind = torch.topk(val_result.flatten(), 20)
                    val_top2_coords = self.get_top2_from_topk(topk_valres_prob, topk_valres_ind)

                    self.viz_valpatch_gt = self.result_painter(self.viz_valpatch_gt, _ids_to_vec(val_gt_coords), dist=gt_dists[gt_shell % len(gt_dists)])
                    self.viz_valpatch = self.result_painter(self.viz_valpatch, _ids_to_vec(val_top2_coords), dist=gt_dists[gt_shell % len(gt_dists)])

                    topk_fpres_prob, topk_fpres_ind = torch.topk(self.fake_B[0, gt_shell, :], 20)
                    fp_top2_coords = self.get_top2_from_topk(topk_fpres_prob, topk_fpres_ind)
                    self.viz_first_patch = self.result_painter(self.viz_first_patch, _ids_to_vec(fp_top2_coords), dist=gt_dists[gt_shell % len(gt_dists)])

                    # can only draw firstpatch gt directions if it's a centerline patch
                    if firstpatch_is_on_cl:
                        fpgt_prob = firstpatch_gt_cpu[fp_gt_loc]
                        fp_gt_coords = self.get_top2_from_topk(torch.as_tensor(fpgt_prob), torch.as_tensor(fp_gt_loc))
                        self.viz_first_patch_gt = self.result_painter(self.viz_first_patch_gt, _ids_to_vec(fp_gt_coords), dist=gt_dists[gt_shell % len(gt_dists)])

        self.opt.phase = prevphase

    def get_top2_from_topk(self, prob_fwd, ind_fwd,
                           prob_bwd=None, ind_bwd=None):
        """Return two direction indices.

        * Legacy mode (no --independent_dir):
          ───────────────────────────────────
          pass a single (prob, ind) pair ⇒ we pick the two furthest-apart
          bins **within the same histogram** (old heuristic).

        * Independent-direction mode:
          ───────────────────────────
          pass **both** forward and backward (prob, ind) pairs.
          We then take *one* bin from each histogram – namely the arg-max
          of each – and return them in the order (forward, backward)."""
        # helper to guarantee vector shape
        def _as_vec(x):
            return x.view(1) if x.dim() == 0 else x

        prob_fwd = _as_vec(prob_fwd);  ind_fwd  = _as_vec(ind_fwd)
        if prob_bwd is not None:
            prob_bwd = _as_vec(prob_bwd);  ind_bwd = _as_vec(ind_bwd)
        # independent-direction path
        if prob_bwd is not None:                     # flag on
            i = torch.argmax(prob_fwd)
            j = torch.argmax(prob_bwd)
            return torch.stack([ind_fwd[i], ind_bwd[j]])

        # legacy path (pick two furthest bins from one histogram)
        prob = prob_fwd.clone()
        i = torch.argmax(prob)
        prob[i] = 0
        j = torch.argmax(prob)
        return torch.stack([ind_fwd[i], ind_fwd[j]])

    def triplanar_canvas(self, patch3d, patch3d_g=None, patch3d_b=None):
        """convert 3d patch to triplanar 2d color image. Optional arguments for separate rgb channels"""
        pw = patch3d.shape[0]
        viz_painting = np.zeros((pw * 2, pw * 2, 3))

        patch3d_r = patch3d
        if patch3d_g is None:
            patch3d_g = patch3d
        if patch3d_b is None:
            patch3d_b = patch3d

        viz_painting[:pw, :pw, 0] = patch3d_r[:, :, pw // 2]
        viz_painting[:pw, :pw, 1] = patch3d_g[:, :, pw // 2]
        viz_painting[:pw, :pw, 2] = patch3d_b[:, :, pw // 2]

        viz_painting[pw:2*pw, :pw, 0] = patch3d_r[:, pw // 2, :]
        viz_painting[pw:2*pw, :pw, 1] = patch3d_g[:, pw // 2, :]
        viz_painting[pw:2*pw, :pw, 2] = patch3d_b[:, pw // 2, :]

        viz_painting[:pw, pw:2*pw, 0] = patch3d_r[pw // 2, :, :]
        viz_painting[:pw, pw:2*pw, 1] = patch3d_g[pw // 2, :, :]
        viz_painting[:pw, pw:2*pw, 2] = patch3d_b[pw // 2, :, :]

        viz_painting = np.clip(viz_painting*64, 0, 255)
        return viz_painting

    def result_painter(self, tp_canvas, rel_coords, dist=2, gval=0):
        """
        TODO: test this
        paint colors on canvas
        """
        pw = tp_canvas.shape[0] // 2

        # draw dots in the middle
        tp_canvas[int(pw * 0.5), int(pw * 0.5), :] = 255  # z
        tp_canvas[int(pw * 1.5), int(pw * 0.5), :] = 255  # y
        tp_canvas[int(pw * 0.5), int(pw * 1.5), :] = 255  # x

        for coord in rel_coords:
            xoff = coord[0]
            yoff = coord[1]
            zoff = coord[2]

            colz = [max(0, x*255) for x in [-zoff, gval, zoff]]
            coly = [max(0, x*255) for x in [-yoff, gval, yoff]]
            colx = [max(0, x*255) for x in [-xoff, gval, xoff]]
            tp_canvas[int(pw * 0.5 + xoff*dist), int(pw * 0.5 + yoff*dist), :] = colz
            tp_canvas[int(pw * 1.5 + xoff*dist), int(pw * 0.5 + zoff*dist), :] = coly
            tp_canvas[int(pw * 0.5 + yoff*dist), int(pw * 1.5 + zoff*dist), :] = colx

        return tp_canvas

    def compute_top2_meandists(self, pred_hist, gt_hist):
        """
        pred_hist, gt_hist :  [N, 500] torch tensors (any device)

        Returns the mean Euclidean distance (float, **CPU python scalar**)
        between the two top-2 direction vectors of the prediction and
        the ground truth histograms.
        """
        # make sure we’re dealing with tensors
        if not torch.is_tensor(pred_hist):
            pred_hist = torch.as_tensor(pred_hist, device=self.device)
        if not torch.is_tensor(gt_hist):
            gt_hist = torch.as_tensor(gt_hist, device=self.device)

        # detach → we don’t need gradients for a metric
        pred_hist = pred_hist.detach()
        gt_hist   = gt_hist.detach()

        # pick the two highest-probability bins for every patch
        # shape:  [N, 2]
        _, pred_top2 = torch.topk(pred_hist, k=2, dim=1)
        _, gt_top2   = torch.topk(gt_hist,   k=2, dim=1)

        # convert spherical-bin id → 3-D unit vector
        # cached lookup table on the correct device
        if not hasattr(self, "_dir_lookup") or self._dir_lookup.device != pred_hist.device:
            # vertices: (500, 3) numpy → torch once
            verts = torch.as_tensor(self.vertices, dtype=torch.float32, device=pred_hist.device)
            self._dir_lookup = verts / torch.linalg.norm(verts, dim=1, keepdim=True)

        # gather vectors, shape [N, 2, 3]
        pred_vecs = self._dir_lookup[pred_top2]
        gt_vecs   = self._dir_lookup[gt_top2]

        # pairwise distances per patch (broadcast over last dim)
        # We want ‖pred[i] – gt[i]‖ for i = 0,1
        dists = torch.linalg.norm(pred_vecs - gt_vecs, dim=2)   # [N, 2]

        # mean over both directions, then over all patches ⇒ scalar
        mean_dist = dists.mean().item()     # .item() → python float on CPU
        return mean_dist

    def compute_topk_meandists(self, pred, goal, k=10):
        """ Compute the mean (euclidean) distance between the top k predictions and any target """
        alldists = []
        for sample in range(pred.shape[0]):
            _, topk_fakeb_ind = torch.topk(pred[sample, :], k)
            _, topk_realb_ind = torch.topk(goal[sample, :], 2)

            indfakeb = topk_fakeb_ind.cpu().numpy()
            indrealb = topk_realb_ind.cpu().numpy()
            topk_fakeb_coords = self.vertices[indfakeb]
            target_coords = self.vertices[indrealb]
            targetdists = []
            for target_coord in target_coords:
                targetdists.append(np.linalg.norm(topk_fakeb_coords - target_coord, axis=1))
            mindists = np.stack(targetdists).min(axis=0)
            alldists.append(np.mean(mindists.flatten()))
        meandist = np.mean(alldists)
        return meandist
   
    def dir_from_spherical(self, logits, frame='patch'):
        """
        Convert a 500-way (or N-way) spherical classification output into
        a unit direction vector.

        Parameters
        ----------
        logits : torch.Tensor
            Shape  [B,   Nclasses]          for a single GT-shell,  **or**
                [B, S, Nclasses]          when you predict several GT-shells.
            Either logits **or** probabilities are fine – we soft-max internally.

        frame : 'patch' | 'world'
            If 'patch' (default) we return the vector in the *local patch* frame.
            If 'world' we rotate it with the per-sample rotation matrix so that
            it lives in world (mm) coordinates.

        Returns
        -------
        vec : torch.Tensor
            Shape [B,   3]  (single shell)  or
                [B, S, 3] (multi-shell)   – always L2-normalised.
        """
        # 1) vertices on the unit sphere  –  shape [Nclasses, 3]
        V = torch.as_tensor(self.vertices,
                            dtype=logits.dtype,
                            device=logits.device)              # (500,3)

        # 2) soft-max over the class dimension
        probs = F.softmax(logits, dim=-1)                      # same shape as `logits`

        # 3) expectation ∑ pᵢ vᵢ
        vec = torch.matmul(probs, V)                           # (B[,S],3)

        # 4) normalise to unit length
        vec = F.normalize(vec, dim=-1, eps=1e-6)

        # 5) optional patch→world conversion
        if frame == 'world':
            # `self.rotM` is the 4 × 4 patch→world matrix saved earlier
            R = self.rotM[:, :3, :3]                           # (B,3,3)
            if vec.dim() == 3:                                 # multi-shell
                R = R.unsqueeze(1)                             # (B,1,3,3) broadcast
            vec = torch.einsum('bij,...j->...i', R, vec)

        return vec

    # @staticmethod
    # @lru_cache(maxsize=128)   # reuse maps when the same volume appears again
    # def _compute_gdt(seg_mask: torch.Tensor,
    #                  seed_voxel: tuple,
    #                  spacing=(1.0, 1.0, 1.0),
    #                  lam: float = 1.0,
    #                  sigma: float = 1.0) -> torch.Tensor:
    #     """
    #     FastGeodis 3-D geodesic distance transform restricted to `seg_mask`.

    #     Parameters
    #     ----------
    #     seg_mask : torch.BoolTensor [D,H,W]
    #         1 → inside segmentation.
    #     seed_voxel : (z, y, x) tuple
    #     spacing : (dz,dy,dx)  physical voxel size in mm.
    #     lam, sigma : see FastGeodis paper/code.
    #     """
    #     device = seg_mask.device
    #     guidance = (~seg_mask).float().unsqueeze(0)        # walls=1, lumen=0
    #     seed_img = torch.zeros_like(guidance)
    #     z, y, x = seed_voxel
    #     seed_img[0, z, y, x] = 1.0

    #     gdt = fg.generalised_geodesic_distance_3d(
    #         guidance, seed_img,
    #         spacing=torch.tensor(spacing, device=device, dtype=torch.float32),
    #         lam=lam, iter_count=2, sigma=sigma
    #     )[0]                                              # drop channel dim
    #     return gdt                                        # still on GPU
    # # ────────────────────────────────────────────────────────────────────────

    # # example: quick consistency check you can call from your training loop
    # # (does **not** influence loss yet – just logs a scalar so you see it)
    # def gdt_shape_consistency(self, seed_voxels, spacing=(1.,1.,1.)):
    #     """
    #     Computes the fraction of the mask that is reachable from every seed
    #     in the current mini-batch and stores it in self.loss_gdt_coverage so it
    #     is plotted alongside the other losses.

    #     seed_voxels : list[ tuple(z,y,x) ] – one per volume in the batch
    #     """
    #     coverages = []
    #     for b in range(self.real_C.size(0)):
    #         seg = (self.real_C[b,0]==1).bool()            # adjust class id if needed
    #         gdt = self._compute_gdt(seg, seed_voxels[b], spacing)
    #         reachable = (gdt < 1e6)                       # FastGeodis sentinel
    #         coverages.append( reachable.sum().float() / seg.sum().float() )

    #     # expose as a “loss” so it shows up in the usual logger/plots
    #     self.loss_gdt_coverage = torch.stack(coverages).mean()
    #     return self.loss_gdt_coverage.item()


class CenterlineLoss(torch.nn.Module):
    """ Same loss function as used in wolterink2018coronary """
    """FIXME TODO DANGER: implementation error. MSE was supposed to be for last logit only (centerline radius)
    Don't use this"""
    def __init__(self, mse_fac=10):
        super(CenterlineLoss, self).__init__()
        self.MSELoss = torch.nn.MSELoss()
        self.BCELoss = torch.nn.BCELoss()
        self.mse_fac = mse_fac

    def forward(self, inputs, targets):
        loss_bce = self.BCELoss(inputs, targets)
        loss_mse = self.MSELoss(inputs, targets)
        return loss_bce + self.mse_fac * (loss_mse)



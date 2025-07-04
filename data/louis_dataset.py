import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import math
from data.base_dataset import BaseDataset  # your base dataset class

###############################################################################
# VolumeContainer for MOT3D HDF5 Data
###############################################################################
class Mot3dVolumeContainer:
    """
    Container class for a MOT3D volume loaded from an HDF5 file.
    Loads the MRI scan (from dataset "MOT3DBH") and the centerline annotations
    (from group "sint_segs_dense_vox"). Also creates a segmentation mask based on
    the centerlines.
    """
    def __init__(self, opt, imnum, inference=False):
        print('Initializing volume: {}'.format(imnum))
        self.opt = opt
        self.current_epoch = 0
        # Construct the file path (replace placeholder "##" with the volume number)
        self.file_path = os.path.join(opt.dataroot, opt.masterpath.replace("##", f"{imnum:02d}"))
        self.seg_path = os.path.join(opt.dataroot, opt.seg_path.replace("##", f"{imnum:02d}"))
        self.no_mask = False
        print(f"Loading HDF5 file: {self.file_path}")
        
        with h5py.File(self.file_path, "r") as f:
            # Load the MRI scan; assume shape is (T, Y, X, Z)
            dset = f["MOT3DBH"]                       # keep the Dataset object
            annotation_t_slice = dset.attrs["annotation_tslice"]  # ← attribute
            mri_scan = dset[:]                        # now pull the pixel data
            self.spacing = np.array(dset.attrs["spacing"], dtype=np.float64)
            self.origin = np.array(dset.attrs["patient_position"], dtype=np.float64)
            
            # Use the first time slice as the 3D volume (shape: (Y, X, Z))
            self.data = torch.Tensor(mri_scan[annotation_t_slice]).unsqueeze(0)  # add channel dimension
            
            # Normalize using median
            if opt.normalisation == 'median':
                data_median = torch.median(self.data[self.data > 10])
                self.data = self.data / data_median
                self.data = torch.clamp(self.data, 0, opt.clip_norm)
            else:
                print("Only median normalisation is supported.")
                exit(1)
            
            # Initialize centerline dictionaries
            self.sint_segs_dense_vox = {}
            self.sint_seg_lens = {}
            self.sint_seg_totlen = 0
            
            # Define the centerline group(s)
            centerline_groups = ["sint_segs_dense_vox"]
            label = 1
            for group_name in centerline_groups:
                if group_name in f:
                    group = f[group_name]
                    for dataset_name in group.keys():
                        centerline = group[dataset_name][:]  # (N, 3) centerline points
                        voxel_indices = centerline.astype(int)
                        # Save centerline coordinates (key as string)
                        self.sint_segs_dense_vox[str(label)] = voxel_indices
                        # Mark centerline points in the mask with the label
                        label += 1
            
            # pick the first point of the first centre-line as the seed
            first_segkey = sorted(self.sint_segs_dense_vox.keys())[0]
            self.seed_voxel = self.sint_segs_dense_vox[first_segkey][0]  # (z,y,x)
            print(self.seed_voxel)

            seg_img = nib.load(self.seg_path)
            # Get the segmentation data
            seg_data = seg_img.get_fdata().astype(np.uint8)
            # Transpose the segmentation data from (Z, X, Y) to (Y, X, Z)
            seg_data = np.transpose(seg_data, (2, 1, 0)) # == 1 # 0: background, 1: centerline, 2: other
            self.seggt = torch.from_numpy(seg_data).unsqueeze(0)

            if self.no_mask:
                self.seggt = torch.zeros(self.data.shape[1:])

        # Compute total length per centerline and overall (for sample indexing)
        gt_dists = [int(x) for x in opt.gt_distances.split(',')]
        for key in sorted(self.sint_segs_dense_vox.keys()):
            seg_len = max(0, self.sint_segs_dense_vox[key].shape[0] - max(gt_dists) * 2)
            self.sint_seg_lens[key] = seg_len
            self.sint_seg_totlen += seg_len
        
        # Load vertices for converting directions to classification labels.
        nclass = opt.nclass
        self.vertices = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  f'../vertices{nclass}.txt'))
        self.vw = opt.isosample_spacing
        self.pw = opt.patch_size
        self.interp = opt.interp
        self.shape = self.data.shape  # (1, Y, X, Z)

    def set_epoch(self, epoch: int):
        """Called once per training epoch by the Dataset."""
        self.current_epoch = epoch

    def get_patch(self, loc, rotate=True, with_seg=False, loc_patchspace=False, convert_to_mm=False, rot_matrix_override=None):
        """
        Extract a patch from the volume at location `loc` using world coordinates.
        """
        spacing = self.spacing
        pw = self.pw
        vw = self.vw
        
        if loc_patchspace:
            loc = [loc[i] * vw for i in range(len(loc))]
        if convert_to_mm:
            loc = vox_coords_to_mm(loc, spacing)
        
        locx, locy, locz = loc
        if rot_matrix_override is None:
            rot_matrix, inv_matrix = get_multiax_rotation_matrix(rotate, iters=3)
        else:
            rot_matrix = rot_matrix_override
            inv_matrix = np.linalg.inv(rot_matrix_override)
            rotate = False                        # keep bookkeeping tidy

        im_patch = draw_sample_4D_world_fast(self.data, locx, locy, locz, spacing,
                                             np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                             rot_matrix, interpolation=self.interp).float()
        im_patch = im_patch.clamp(0, self.opt.clip_norm)
        
        if with_seg:
            if self.no_mask:
                seggt_patch = draw_sample_3D_world_fast(self.seggt, locx, locy, locz, spacing,
                                                        np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                                        rot_matrix, interpolation='nearest').float()
            else:
                seggt_patch = draw_sample_4D_world_fast(self.seggt, locx, locy, locz, spacing,
                                                    np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                                    rot_matrix, interpolation='nearest').float()
            return im_patch, seggt_patch, rot_matrix, inv_matrix
        else:
            return im_patch, rot_matrix, inv_matrix

    
    def get_sample(self, intest_num, intest_index, valvol=False):
        """
        Sample a patch centered on a point along a centerline.
        Returns: image patch, target direction labels, segmentation patch.
        """
        gt_dists = [int(x) for x in self.opt.gt_distances.split(',')]
        spacing = self.spacing
        vertices = self.vertices
        pw = self.pw
        vw = self.vw
        gt_spacing = self.opt.orig_gt_spacing
        displace_sigma = 0.0 if valvol else self.opt.displace_augmentation_mm
        gt_sigma = self.opt.gt_sigma
        rotate = False if valvol else self.opt.rotation_augmentation
        
        # Get the centerline (in voxel space) from the dictionary.
        intest_coords_vox = self.sint_segs_dense_vox[str(intest_num)]
        intest_coords_mm = vox_coords_to_mm(intest_coords_vox, spacing)
        
        locx = intest_coords_mm[intest_index, 0]
        locy = intest_coords_mm[intest_index, 1]
        locz = intest_coords_mm[intest_index, 2]
        point_orig = np.reshape(np.array([locx, locy, locz]), (1, 3))
        
        if self.opt.isTrain:
            # Apply a random displacement (augmentation)
            diff_coord = intest_coords_mm[intest_index - 1, :] - intest_coords_mm[intest_index, :]
            normvec = np.cross(diff_coord, np.random.random((1, 3)))[0]
            normvec = normvec / (np.linalg.norm(normvec) + 1e-6)
            locx += np.random.normal(0, displace_sigma, 1) * normvec[0]
            locy += np.random.normal(0, displace_sigma, 1) * normvec[1]
            locz += np.random.normal(0, displace_sigma, 1) * normvec[2]
        
        if self.opt.only_displace_input:
            point = point_orig
        else:
            point = np.reshape(np.array([locx, locy, locz]), (1, 3))
        
        rot_matrix, inv_matrix = get_multiax_rotation_matrix(rotate, iters=3)
        im_patch = draw_sample_4D_world_fast(self.data, locx, locy, locz, spacing,
                                             np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                             rot_matrix, interpolation=self.interp).float()
        if self.no_mask:
            seggt_patch = draw_sample_3D_world_fast(self.seggt, locx, locy, locz, spacing,
                                        np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                        rot_matrix, interpolation='nearest').float()
        else:
            seggt_patch = draw_sample_4D_world_fast(self.seggt, locx, locy, locz, spacing,
                                                    np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                                    rot_matrix, interpolation='nearest').float()
            
        target_list = []
        for gt_dist in gt_dists:
            nextp = intest_coords_mm[intest_index - gt_dist, :]
            prevp = intest_coords_mm[intest_index + gt_dist, :]
            
            displacement_forward = (nextp - point)
            displacement_forward = gt_spacing * gt_dist * displacement_forward / (np.linalg.norm(displacement_forward) + 1e-6)
            displacement_back = (prevp - point)
            displacement_back = gt_spacing * gt_dist * displacement_back / (np.linalg.norm(displacement_back) + 1e-6)
            if self.opt.independent_dir:
                # get the ground-truth backward unit vector
                prev_vec_world = -displacement_back.reshape(3)
                prev_vec_world /= np.linalg.norm(prev_vec_world) + 1e-6

                # inject Gaussian noise in *world* coords
                sigma = self._current_sigma_rad()          # already in radians
                if sigma > 0:
                    prev_vec_world += np.random.normal(0, sigma, size=3)
                    prev_vec_world /= np.linalg.norm(prev_vec_world) + 1e-6   # re-norm

                # rotate into the local patch frame
                prev_vec_local = inv_matrix[:3, :3] @ prev_vec_world   # NumPy @ NumPy

                # broadcast to three constant channels and append
                dir_map = torch.from_numpy(prev_vec_local.astype(np.float32)) \
                            .view(3, 1, 1, 1)                       # shape (3,1,1,1)
                dir_map = dir_map.expand(-1, pw, pw, pw)              # (3, pw, pw, pw)
                im_patch = torch.cat([im_patch, dir_map], dim=0)      # +3 channels

                tgt_fwd = self.directionToClass(vertices,
                                                displacement_forward,
                                                rotMatrix=inv_matrix,
                                                sigma=gt_sigma)
                target_list.append(torch.as_tensor(tgt_fwd,  dtype=torch.float32))

                tgt_bwd = self.directionToClass(vertices,
                                                displacement_back,
                                                rotMatrix=inv_matrix,
                                                sigma=gt_sigma)
                target_list.append(torch.as_tensor(tgt_bwd, dtype=torch.float32))

            else:
                target = self.directionToClass(vertices, displacement_forward, rotMatrix=inv_matrix, sigma=gt_sigma)
                target += self.directionToClass(vertices, displacement_back, rotMatrix=inv_matrix, sigma=gt_sigma)

                target = (target / np.sum(target))
                target_list.append(torch.as_tensor(target).float())

        target = torch.stack(target_list, dim=0)
        im_patch = im_patch.clamp(0, self.opt.clip_norm)
        center_mm = torch.tensor([float(locx), float(locy), float(locz)])
        rotM = torch.from_numpy(rot_matrix.astype(np.float32))
        if valvol:
            # validation crops don’t need metadata
            return im_patch, target, seggt_patch
        else:
            return im_patch, target, seggt_patch, center_mm, rotM

    # def _current_sigma_rad(self):
    #     """
    #     Returns the current σ (radians) for the Gaussian noise that corrupts the
    #     previous-direction cue.  During validation or when --prev_dir_noise_max=0
    #     this is guaranteed to be 0.0.
    #     """
    #     # not training? -> no noise
    #     if not getattr(self.opt, 'isTrain', False):
    #         return 0.0

    #     # no warm-up requested? -> constant σ
    #     if self.opt.prev_dir_noise_warmup == 0:
    #         return np.deg2rad(float(self.opt.prev_dir_noise_max))

    #     # linear warm-up over epochs
    #     frac = min(1.0, self.current_epoch / float(self.opt.prev_dir_noise_warmup))
    #     return np.deg2rad(frac * float(self.opt.prev_dir_noise_max))

    def _current_sigma_rad(self):
        """Exponential warm-up for σ (radians)."""
        if not getattr(self.opt, 'isTrain', False):
            return 0.0

        max_sigma  = np.deg2rad(float(self.opt.prev_dir_noise_max))
        warmup_ep  = max(1, int(self.opt.prev_dir_noise_warmup))   # avoid div/0

        # β chosen s.t. σ ≈ 0.95·σ_max at epoch == warmup_ep
        beta = 3.0 / warmup_ep

        # epochs are 0-based internally → add 1 so that epoch0 == 1
        e = self.current_epoch + 1

        sigma = max_sigma * (1.0 - math.exp(-beta * e))
        return min(sigma, max_sigma)          # clamp in case of very large e

    def directionToClass(self, vertices, target, rotMatrix=np.eye(4, dtype='float32'), sigma=0):
        """
        Convert a displacement vector into a distribution over sphere vertices.
        """
        vertexlength = np.linalg.norm(np.squeeze(vertices[0, :]))
        target = target.reshape((1, 3))
        target = np.dot(rotMatrix, np.array([target[0, 0], target[0, 1], target[0, 2], 0.0]))
        target = target[:3]
        target = target / (np.linalg.norm(target) / vertexlength)
        
        dist_to_vert = np.linalg.norm(vertices - target, axis=1)
        distro = np.zeros(dist_to_vert.shape, dtype='float32')
        if sigma == 0:
            distro[np.argmin(dist_to_vert)] = 1.0
        else:
            distro = np.clip(sigma - dist_to_vert, 0, sigma)
        return distro

###############################################################################
# Dataset Class
###############################################################################
class louisDataset(BaseDataset):
    """
    A dataset class that aggregates MOT3D volumes.
    It provides samples consisting of an image patch, ground-truth direction targets,
    segmentation GT (the centerline mask), and a flag indicating whether the patch
    was sampled on a centerline.
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.volnums = [] if opt.trainvols == '-1' else sorted([int(x) for x in opt.trainvols.split(',')])
        self.valnum = opt.validationvol

        self.current_epoch = 0
        
        # Build volume containers for training volumes.
        self.volumes = {}
        for volnum in self.volnums:
            self.volumes[volnum] = Mot3dVolumeContainer(opt, volnum)
        
        # Build a volume container for validation.
        self.valvol = Mot3dVolumeContainer(opt, self.valnum, inference=True)
        self.valvol.valvol = True
        self.valbatch_nums = []
        
        np.random.seed(42)
        for i in range(opt.batch_size):
            intestlen = 0
            while intestlen < 1:
                intestnum = np.random.choice(list(self.valvol.sint_segs_dense_vox.keys()))
                intestlen = self.valvol.sint_seg_lens[intestnum]
            intestloc = np.random.randint(intestlen)
            self.valbatch_nums.append((int(intestnum), intestloc))

    def set_epoch(self, epoch: int):
        """Tell the dataset which training epoch we are in."""
        self.current_epoch = epoch
        # propagate to all training volumes
        for vol in self.volumes.values():      # self.volumes is the dict you build in __init__
            if hasattr(vol, "set_epoch"):
                vol.set_epoch(epoch)

        # and to the validation volume
        if hasattr(self, "valvol") and hasattr(self.valvol, "set_epoch"):
            self.valvol.set_epoch(epoch)

    def get_valdata(self):
        patches, targets, segtargets = [], [], []
        for num, loc in self.valbatch_nums:
            patch, target, segtarget = self.valvol.get_sample(num, loc, valvol=True)
            patches.append(patch)
            targets.append(target)
            segtargets.append(segtarget)
        valbatch = torch.stack(patches, dim=0)
        valbatch_gt = torch.stack(targets, dim=0)
        valbatch_seggt = torch.stack(segtargets, dim=0)
        return valbatch, valbatch_gt, valbatch_seggt, self.valvol.vertices
    
    def get_gt_by_index(self, index):
        index_counter = index
        for volnum in self.volnums:
            vol = self.volumes[volnum]
            if index_counter >= vol.sint_seg_totlen:
                index_counter -= vol.sint_seg_totlen
            else:
                local_counter = index_counter
                for seg in sorted(vol.sint_segs_dense_vox.keys()):
                    if local_counter >= vol.sint_seg_lens[seg]:
                        local_counter -= vol.sint_seg_lens[seg]
                    else:
                        int_num = seg
                        int_index = local_counter + max([int(x) for x in self.opt.gt_distances.split(',')])
                        return volnum, int_num, int_index
        print('Index {} out of range!'.format(index))
        exit(1)
    
    def __getitem__(self, index):
        """
        Return a dictionary with:
          - 'A': image patch (torch.Tensor)
          - 'B': ground-truth direction targets (torch.Tensor)
          - 'C': segmentation ground truth (centerline mask patch)
          - 'D': Boolean flag (True if the patch was sampled on a centerline)
        """
        # With probability non_centerline_ratio (if training) we sample a random patch.
        centerline_patch = not (self.opt.isTrain and np.random.rand() < self.opt.non_centerline_ratio)
        volnum, int_num, int_index = self.get_gt_by_index(index)
        if centerline_patch:
            imdata, gtdata, seggtdata, center_mm, rotM = self.volumes[volnum].get_sample(int_num, int_index)
        else:
            patchvol = self.volumes[volnum]
            # Randomly choose a location in the volume (x, y, z order)
            patchloc = (np.random.rand() * patchvol.shape[2],
                        np.random.rand() * patchvol.shape[1],
                        np.random.rand() * patchvol.shape[0])
            imdata, seggtdata, rotmat, irotmat = patchvol.get_patch(patchloc, with_seg=True)
            center_mm = torch.tensor(vox_coords_to_mm(patchloc, patchvol.spacing), dtype=torch.float32)
            rotM = torch.as_tensor(rotmat, dtype=torch.float32)

            if self.opt.independent_dir:
                n_shells = len(self.opt.gt_distances.split(',')) * 2
                gtdata = torch.ones((n_shells, 500), dtype=torch.float32) * -1
                pw = self.pw
                dir_map = torch.zeros(3, pw, pw, pw, dtype=imdata.dtype)
                imdata  = torch.cat([imdata, dir_map], dim=0)
            else:
                gtdata = torch.ones((1, 500)) * -1
        return {
            'A': imdata,
            'B': gtdata,
            'C': seggtdata,
            'D': centerline_patch,
            'center_mm': center_mm,
            'rotM': rotM,
            'vol_id': volnum,
            
            # 'seed_voxels': torch.as_tensor(self.seed_voxel, dtype=torch.long),
            # 'spacing':   torch.as_tensor(self.spacing, dtype=torch.float32)
        }
    
    def __len__(self):
        tot_samples = 0
        for volnum in self.volnums:
            tot_samples += np.sum(list(self.volumes[volnum].sint_seg_lens.values()))
        return tot_samples
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--masterpath', type=str,
                            default='/MOT3D_multi_tslice_MII##a.hdf5',
                            help='Name of the HDF5 dataset file (## is replaced by volume number)')
        parser.add_argument('--seg_path', type=str,
                            default='masked_segmentations/MII##a_seg_t2_5c.nii',
                            help='Folder containing the segmentation NIfTI files')
        parser.add_argument('--normalisation', type=str, default='median', help='normalisation strategy. [median | mean | max | none]')
        parser.add_argument('--trainvols', type=str, default='1,2,3', help='training patient numbers, comma separated')
        parser.add_argument('--validationvol', type=int, default=5, help='validation patient number')
        parser.add_argument('--clip_norm', type=int, default=4, help='maximum clamp value after normalisation')
        parser.add_argument('--nclass', type=int, default=500, help='number of classes on the sphere')
        parser.add_argument('--patch_size', type=int, default=19, help='patch size')
        parser.add_argument('--tslice_start', type=int, default=2, help='first tslice (twidth equals input_nc)')
        parser.add_argument('--value_augmentation', type=float, default=0, help='image scaling augmentation factor width (0 for disable)')
        parser.add_argument('--gt_distances', type=str, default='3', help='GT distances, comma separated')
        parser.add_argument('--isosample_spacing', type=float, default=1.0, help='iso sampling spacing in mm')
        parser.add_argument('--orig_gt_spacing', type=float, default=0.5, help='resolution of ground truth points in mm')
        parser.add_argument('--displace_augmentation_mm', type=float, default=1.0, help='displacement augmentation in mm')
        parser.add_argument('--rotation_augmentation', type=bool, default=True, help='rotation augmentations')
        parser.add_argument('--gt_sigma', type=float, default=0, help='sigma for GT smoothing')
        parser.add_argument('--interp', type=str, default='linear', help='interpolation method: linear or nearest')
        parser.add_argument('--only_displace_input', action='store_true', help='do not recompute gt angle')
        parser.add_argument('--seg_bce_factor', type=float, default=1, help='factor of seg BCE loss component')
        parser.add_argument('--selfconsistency_factor', type=float, default=0, help='factor selfconsistency loss component')
        parser.add_argument('--selfconsistency_range', type=int, default=4, help='length of selfconsistency rays')
        parser.add_argument('--selfconsistency_delay', type=int, default=16, help='delay selfconsistency in epochs')
        parser.add_argument('--dir_bce_factor', type=float, default=100, help='factor of direction BCE loss component')
        parser.add_argument('--dir_bce_offset', type=float, default=1, help='offset for nicer plotting')
        parser.add_argument('--independent_dir', action='store_true', help='predict forward and backward directions separately')
        parser.add_argument('--prev_dir_noise_max',   type=float, default=35,
                    help='max σ for previous-direction noise [deg]')
        parser.add_argument('--prev_dir_noise_warmup', type=int,   default=40,
                            help='epochs over which σ grows linearly')
        return parser

###############################################################################
# Helper Functions (Coordinate conversions, rotation, interpolation, etc.)
###############################################################################
def vox_coords_to_mm(coords, spacing):
    if np.size(coords) == 3:
        return [coords[i] * spacing[i] for i in range(3)]
    coords_mm = []
    for coord in coords:
        coords_mm.append([coord[i] * spacing[i] for i in range(3)])
    return np.array(coords_mm)

def mm_coords_to_vox(coords, spacing):
    if np.size(coords) == 3:
        return [coords[i] / spacing[i] for i in range(3)]
    coords_vox = []
    for coord in coords:
        coords_vox.append([coord[i] / spacing[i] for i in range(3)])
    return np.array(coords_vox)

def get_multiax_rotation_matrix(rotate=True, iters=3):
    if not rotate:
        return np.eye(4, dtype='float32'), np.eye(4, dtype='float32')
    rotmats = []
    invrotmats = []
    for i in range(iters):
        rot, inv_rot = get_rotation_matrix(rotate)
        rotmats.append(rot)
        invrotmats.insert(0, inv_rot)
    rot = rotmats[0]
    inv_rot = invrotmats[0]
    for i in range(1, len(rotmats)):
        rot = rot @ rotmats[i]
        inv_rot = inv_rot @ invrotmats[i]
    return rot, inv_rot

def get_rotation_matrix(rotate=True):
    if not rotate:
        return np.eye(4, dtype='float32'), np.eye(4, dtype='float32')
    else:
        rotangle = np.random.randint(0, 3, 1)
        alpha = beta = gamma = 0.0
        if rotangle == 0:
            alpha = (np.squeeze(np.float64(np.random.randint(0, 360, 1))) / 180.) * np.pi
        if rotangle == 1:
            beta = (np.squeeze(np.float64(np.random.randint(0, 360, 1))) / 180.) * np.pi
        if rotangle == 2:
            gamma = (np.squeeze(np.float64(np.random.randint(0, 360, 1))) / 180.) * np.pi
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        rotMatrix = np.array([[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg, 0.],
                              [cb * sg, ca * cg + sa * sb * sg, -cg * sa + ca * sb * sg, 0.],
                              [-sb,     cb * sa,               ca * cb,                0],
                              [0, 0, 0, 1.]], dtype='float32')
        # Inverse rotation uses negative angles
        alpha, beta, gamma = -alpha, -beta, -gamma
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        invMatrix = np.array([[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg, 0.],
                              [cb * sg, ca * cg + sa * sb * sg, -cg * sa + ca * sb * sg, 0.],
                              [-sb,     cb * sa,               ca * cb,                0],
                              [0, 0, 0, 1.]], dtype='float32')
        return rotMatrix, invMatrix

def draw_sample_4D_world_fast(image, x, y, z, imagespacing, patchsize, patchspacing,
                              rot_matrix=np.eye(4, dtype='float32'), interpolation='nearest'):
    patchmargin = (patchsize - 1) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    xs = (x + (unra[0] - patchmargin[0]) * patchspacing[0]) / imagespacing[0]
    ys = (y + (unra[1] - patchmargin[1]) * patchspacing[1]) / imagespacing[1]
    zs = (z + (unra[2] - patchmargin[2]) * patchspacing[2]) / imagespacing[2]
    xs = xs - (x / imagespacing[0])
    ys = ys - (y / imagespacing[1])
    zs = zs - (z / imagespacing[2])
    coords = np.concatenate((np.reshape(xs, (1, xs.shape[0])), 
                               np.reshape(ys, (1, ys.shape[0])),
                               np.reshape(zs, (1, zs.shape[0])), 
                               np.zeros((1, xs.shape[0]), dtype='float32')), axis=0)
    coords = np.dot(rot_matrix, coords)
    xs = np.squeeze(coords[0, :]) + (x / imagespacing[0])
    ys = np.squeeze(coords[1, :]) + (y / imagespacing[1])
    zs = np.squeeze(coords[2, :]) + (z / imagespacing[2])
    if interpolation == 'linear':
        patch = fast_trilinear_4d(image, xs, ys, zs)
    else:
        patch = fast_nearest_4d(image, xs, ys, zs)
    reshaped_patch = patch.view([image.shape[0]] + patchsize.tolist())
    return reshaped_patch

def draw_sample_3D_world_fast(image, x, y, z, imagespacing, patchsize, patchspacing,
                              rot_matrix=np.eye(4, dtype='float32'), interpolation='nearest'):
    patchmargin = (patchsize - 1) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    xs = (x + (unra[0] - patchmargin[0]) * patchspacing[0]) / imagespacing[0]
    ys = (y + (unra[1] - patchmargin[1]) * patchspacing[1]) / imagespacing[1]
    zs = (z + (unra[2] - patchmargin[2]) * patchspacing[2]) / imagespacing[2]
    xs = xs - (x / imagespacing[0])
    ys = ys - (y / imagespacing[1])
    zs = zs - (z / imagespacing[2])
    coords = np.concatenate((np.reshape(xs, (1, xs.shape[0])), 
                               np.reshape(ys, (1, ys.shape[0])),
                               np.reshape(zs, (1, zs.shape[0])), 
                               np.zeros((1, xs.shape[0]), dtype='float32')), axis=0)
    coords = np.dot(rot_matrix, coords)
    xs = np.squeeze(coords[0, :]) + (x / imagespacing[0])
    ys = np.squeeze(coords[1, :]) + (y / imagespacing[1])
    zs = np.squeeze(coords[2, :]) + (z / imagespacing[2])
    if interpolation == 'linear':
        patch = fast_trilinear(image, xs, ys, zs)
    else:
        patch = fast_nearest(image, xs, ys, zs)
    reshaped_patch = patch.view(patchsize.tolist())
    return reshaped_patch

def fast_trilinear_4d(input_array, x_indices, y_indices, z_indices):
    # Convert indices to integers
    x0 = x_indices.astype(np.int32)
    y0 = y_indices.astype(np.int32)
    z0 = z_indices.astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Clamp indices that exceed the maximum dimension
    x0[np.where(x0 >= input_array.shape[1])] = input_array.shape[1] - 1
    y0[np.where(y0 >= input_array.shape[2])] = input_array.shape[2] - 1
    z0[np.where(z0 >= input_array.shape[3])] = input_array.shape[3] - 1
    x1[np.where(x1 >= input_array.shape[1])] = input_array.shape[1] - 1
    y1[np.where(y1 >= input_array.shape[2])] = input_array.shape[2] - 1
    z1[np.where(z1 >= input_array.shape[3])] = input_array.shape[3] - 1

    # **Clamp negative indices to 0**
    x0[np.where(x0 < 0)] = 0
    y0[np.where(y0 < 0)] = 0
    z0[np.where(z0 < 0)] = 0
    x1[np.where(x1 < 0)] = 0
    y1[np.where(y1 < 0)] = 0
    z1[np.where(z1 < 0)] = 0

    # Calculate the fractional parts
    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    # Convert arrays to torch tensors
    x = torch.as_tensor(x)
    x0 = torch.as_tensor(x0)
    x1 = torch.as_tensor(x1)
    y = torch.as_tensor(y)
    y0 = torch.as_tensor(y0)
    y1 = torch.as_tensor(y1)
    z = torch.as_tensor(z)
    z0 = torch.as_tensor(z0)
    z1 = torch.as_tensor(z1)

    # Pad the input array (if necessary)
    input_array = F.pad(input_array, (0, 1, 0, 1, 0, 1, 0, 0), mode='constant')
    
    single_t_patches = []
    for t in range(input_array.shape[0]):
        single_t_array = input_array[t]
        single_t_patch = (single_t_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z) +
                          single_t_array[x1, y0, z0] * x * (1 - y) * (1 - z) +
                          single_t_array[x0, y1, z0] * (1 - x) * y * (1 - z) +
                          single_t_array[x0, y0, z1] * (1 - x) * (1 - y) * z +
                          single_t_array[x1, y0, z1] * x * (1 - y) * z +
                          single_t_array[x0, y1, z1] * (1 - x) * y * z +
                          single_t_array[x1, y1, z0] * x * y * (1 - z) +
                          single_t_array[x1, y1, z1] * x * y * z)
        single_t_patches.append(single_t_patch)
    output = torch.stack(single_t_patches)
    return output


def fast_nearest_4d(input_array, x_indices, y_indices, z_indices):
    """
    Modified Jelmer's code from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    Uses full first dimension (t slicing should happen before input array is supplied)
    This version zero-pads
    """
    if len(input_array.shape) != 4:
        print('input array to nearest_4d is not 4D!!')
        1 / 0

    x_ind = (x_indices + 0.5).astype(np.integer)
    y_ind = (y_indices + 0.5).astype(np.integer)
    z_ind = (z_indices + 0.5).astype(np.integer)

    x_ind[np.where(x_ind >= input_array.shape[1])] = input_array.shape[1]
    y_ind[np.where(y_ind >= input_array.shape[2])] = input_array.shape[2]
    z_ind[np.where(z_ind >= input_array.shape[3])] = input_array.shape[3]
    x_ind[np.where(x_ind < 0)] = input_array.shape[1]
    y_ind[np.where(y_ind < 0)] = input_array.shape[2]
    z_ind[np.where(z_ind < 0)] = input_array.shape[3]
    input_array = F.pad(input_array, (0,1,0,1,0,1,0,0), mode='constant')
    return input_array[:, x_ind, y_ind, z_ind]

def fast_trilinear(input_array, x_indices, y_indices, z_indices):
    x0 = x_indices.astype(np.int32)
    y0 = y_indices.astype(np.int32)
    z0 = z_indices.astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    x0[np.where(x0 >= input_array.shape[0])] = input_array.shape[0] - 1
    y0[np.where(y0 >= input_array.shape[1])] = input_array.shape[1] - 1
    z0[np.where(z0 >= input_array.shape[2])] = input_array.shape[2] - 1
    x1[np.where(x1 >= input_array.shape[0])] = input_array.shape[0] - 1
    y1[np.where(y1 >= input_array.shape[1])] = input_array.shape[1] - 1
    z1[np.where(z1 >= input_array.shape[2])] = input_array.shape[2] - 1
    x0[np.where(x0 < 0)] = 0
    y0[np.where(y0 < 0)] = 0
    z0[np.where(z0 < 0)] = 0
    x1[np.where(x1 < 0)] = 0
    y1[np.where(y1 < 0)] = 0
    z1[np.where(z1 < 0)] = 0
    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0
    x = torch.as_tensor(x)
    x0 = torch.as_tensor(x0)
    x1 = torch.as_tensor(x1)
    y = torch.as_tensor(y)
    y0 = torch.as_tensor(y0)
    y1 = torch.as_tensor(y1)
    z = torch.as_tensor(z)
    z0 = torch.as_tensor(z0)
    z1 = torch.as_tensor(z1)
    input_array = F.pad(input_array, (0, 1, 0, 1, 0, 1), mode='constant')
    output = (input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z) +
              input_array[x1, y0, z0] * x * (1 - y) * (1 - z) +
              input_array[x0, y1, z0] * (1 - x) * y * (1 - z) +
              input_array[x0, y0, z1] * (1 - x) * (1 - y) * z +
              input_array[x1, y0, z1] * x * (1 - y) * z +
              input_array[x0, y1, z1] * (1 - x) * y * z +
              input_array[x1, y1, z0] * x * y * (1 - z) +
              input_array[x1, y1, z1] * x * y * z)
    return output

def fast_nearest(input_array, x_indices, y_indices, z_indices):
    x_ind = (x_indices + 0.5).astype(np.int32)
    y_ind = (y_indices + 0.5).astype(np.int32)
    z_ind = (z_indices + 0.5).astype(np.int32)
    x_ind[np.where(x_ind >= input_array.shape[0])] = input_array.shape[0] - 1
    y_ind[np.where(y_ind >= input_array.shape[1])] = input_array.shape[1] - 1
    z_ind[np.where(z_ind >= input_array.shape[2])] = input_array.shape[2] - 1
    x_ind[np.where(x_ind < 0)] = 0
    y_ind[np.where(y_ind < 0)] = 0
    z_ind[np.where(z_ind < 0)] = 0
    input_array = F.pad(input_array, (0, 1, 0, 1, 0, 1), mode='constant')
    return input_array[x_ind, y_ind, z_ind]


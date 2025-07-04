import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import skimage

from sklearn.cluster import DBSCAN
from skimage.filters import meijering
from data.base_dataset import BaseDataset  # your base dataset class

#############################
#  Helper Function for Preprocessing
#############################

def skeletonize_itk(binary_centerline):
    """
    Applies ITK's BinaryThinningImageFilter to perform true 3D skeletonization.
    :param binary_centerline: 3D numpy array (binary mask of centerline)
    :return: Skeletonized 3D numpy array
    """
    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(binary_centerline.astype(np.uint8))

    # Apply Binary Thinning (ITK's 3D skeletonization)
    thinning_filter = sitk.BinaryThinningImageFilter()
    skeleton_image = thinning_filter.Execute(sitk_image)

    # Convert back to numpy array
    return sitk.GetArrayFromImage(skeleton_image)

def skeletonize_skimage(binary_centerline):
    """
    Applies skimage's skeletonize_3d function to perform true 3D skeletonization.
    :param binary_centerline: 3D numpy array (binary mask of centerline)
    :return: Skeletonized 3D numpy array
    """
    return skimage.morphology.skeletonize(binary_centerline, method='lee')

def apply_meijering_filter(image_3d, sigmas, black_ridges):
    """
    Applies the Meijering filter to enhance tubular structures in a 3D image.
    :param image_3d: 3D numpy array (original MRI scan)
    :return: 3D numpy array (Meijering-enhanced image)
    """
    return meijering(image_3d, sigmas=sigmas, black_ridges=black_ridges)  # Adjust sigmas as needed

def split_centerline(coords, eps=2, min_samples=1):
    """
    Splits centerline coordinates into multiple segments using DBSCAN clustering.
    
    Parameters:
        coords (np.ndarray): Array of shape (N, 3) with centerline coordinates.
        eps (float): Maximum distance between two samples for them to be in the same cluster.
        min_samples (int): Minimum number of points to form a cluster.
        
    Returns:
        List[np.ndarray]: A list of arrays, each representing a connected segment.
    """
    if len(coords) == 0:
        return []
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    
    segments = []
    for label in np.unique(labels):
        if label == -1:  # Optional: ignore noise points
            continue
        segment = coords[labels == label]
        segments.append(segment)
    return segments

def n4_bias_correct_yxz(vol_yxz, iters=50):
    """
    Run N4 bias-field correction on a 3-D volume.
    Down-samples by <shrink> during estimation to keep it fast on cine stacks.
    """
    # 1.  Re-order to (Z, Y, X) for SimpleITK
    vol_zyx = np.transpose(vol_yxz, (2, 0, 1))

    img = sitk.GetImageFromArray(vol_zyx, isVector=False)
    n4  = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([iters])     # one scale = len(list)

    img_corr = n4.Execute(img)                   # same size as input

    # 2.  Back to (Y, X, Z)
    vol_corr_yxz = np.transpose(sitk.GetArrayFromImage(img_corr),
                                (1, 2, 0))
    return vol_corr_yxz.astype(np.float32)

def robust_clip(vol, low_pct=0.5, high_pct=99.5):
    lo, hi = np.percentile(vol, (low_pct, high_pct))
    return np.clip(vol, lo, hi)

def preprocess_mri(vol_yxz):
    """
    Y,X,Z volume  →  N4  →  percentile clip  →  float32 (Y,X,Z)
    """
    vol = n4_bias_correct_yxz(vol_yxz)   # orientation-safe N4
    vol = robust_clip(vol)               # remove fat/air outliers
    return vol.astype(np.float32)

#############################
#  NIFTI-Specific Loading
#############################

def load_mri(scan_path, sigmas, black_ridges):
    """
    Load the MRI scan from a NIFTI file (file with 'MOTILITY' in its name) and apply the Meijering filter.
    Returns the original MRI (after a DICOM-like transpose), the voxel spacings, 
    orientation information, and the Meijering-filtered image.
    """
    mri_files = [f for f in os.listdir(scan_path) if "motility" in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz'))]
    if not mri_files:
        raise FileNotFoundError("No MRI (MOTILITY) file found in folder: " + scan_path)
    mri_file = os.path.join(scan_path, mri_files[0])
    nii = nib.load(mri_file)
    im3d = nii.get_fdata() # (Y, X, Z)
    im3d = preprocess_mri(im3d)
    im3d = np.transpose(im3d, (0, 2, 1))      # (Y, Z, X)  ← last axis = coronal (Y)
    meijering_filtered = apply_meijering_filter(im3d, sigmas, black_ridges)
    zooms = nii.header.get_zooms()[:3]
    ps = [zooms[0], zooms[2], zooms[1]]   # (dY, dZ, dX) = spacing of (Y,Z,X)
    affine = nii.affine
    im_or = affine[:3, :3]
    im_pos = affine[:3, 3]
    shape = im3d.shape
    im_end_pos = nib.affines.apply_affine(affine, (shape[0]-1, shape[1]-1, shape[2]-1))
    return im3d, meijering_filtered, ps, im_or, im_pos, im_end_pos

#############################
#  Volume Container for NIFTI
#############################

class VolumeContainerNIFTI:
    def __init__(self, opt, imnum, inference=False):
        print("Initializing volume for patient: pt{}".format(imnum))
        self.opt = opt
        self.set_root = os.path.join(opt.dataroot, f"pt{imnum}")
        self.no_mask = False
        sigmas = [1]
        black_ridges = False
        im3d, meijering_filtered, ps, im_or, im_pos, im_end_pos = load_mri(self.set_root, sigmas, black_ridges)
        self.data = torch.Tensor(im3d).unsqueeze(0)
        if opt.normalisation != 'median':
            print("Only median normalisation supported")
            quit(1)
        data_median = torch.median(self.data[self.data > 0])
        self.data = self.data / data_median
        self.data = torch.clamp(self.data, 0, opt.clip_norm)

        # Load segmentation (look for file containing "bowel")
        seg_files = [f for f in os.listdir(self.set_root) if "bowel" in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz'))]
        if seg_files:
            seg_file = os.path.join(self.set_root, seg_files[0])
            seg_nii = nib.load(seg_file)
            seg_data = seg_nii.get_fdata()
            seg_data = np.transpose(seg_data, (0, 2, 1))
            # seg_data = np.transpose(seg_data, (1, 0, 2))
            self.seggt = torch.Tensor(seg_data).unsqueeze(0)
        else:
            self.seggt = torch.zeros(self.data.shape[1:])

        # Load centerline data (file with "centerline")
        self.sint_seg_lens = {}
        self.sint_seg_totlen = 0
        gt_dists = [int(x) for x in opt.gt_distances.split(',')]
        if not inference:
            self.sint_segs_dense_vox = {}
            centerline_files = [f for f in os.listdir(self.set_root) if "centerline" in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz'))]
            for i, fname in enumerate(sorted(centerline_files)):
                fullfname = os.path.join(self.set_root, fname)
                print("Loading centerline from:", fullfname)
                centerline_nii = nib.load(fullfname)
                cl_data = centerline_nii.get_fdata()
                binary_centerline = (cl_data > 0).astype(np.uint8)
                cl_skeletonized = skeletonize_skimage(binary_centerline)
                coords = np.column_stack(np.nonzero(cl_skeletonized))
                coords = coords[:, [0, 2, 1]] # (y, x, z)
                coords = coords[coords[:,2].argsort()]
                split_centerlines = split_centerline(coords, eps=2, min_samples=1)
                for j, seg in enumerate(split_centerlines):
                    self.sint_segs_dense_vox[str(j+1)] = seg

            for seg in sorted(self.sint_segs_dense_vox.keys()):
                self.sint_seg_lens[seg] = max(0, self.sint_segs_dense_vox[seg].shape[0] - max(gt_dists)*2)
                self.sint_seg_totlen += self.sint_seg_lens[seg]

        # Load vertices for converting directions to classification labels.
        nclass = opt.nclass
        self.vertices = np.loadtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                  f'../vertices{nclass}.txt'))
        self.vw = opt.isosample_spacing
        self.pw = opt.patch_size
        self.interp = opt.interp
        self.shape = self.data.shape  # (1, Y, X, Z)
        self.spacing = ps

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
                prev_vec_world = -displacement_back.reshape(3)
                prev_vec_world /= (np.linalg.norm(prev_vec_world) + 1e-6)
                prev_vec_local = inv_matrix[:3,:3] @ prev_vec_world
                dir_map = torch.from_numpy(prev_vec_local.astype(np.float32)).view(3,1,1,1)
                dir_map = dir_map.expand(-1, pw, pw, pw)      # pw = patch_size
                im_patch = torch.cat([im_patch, dir_map], dim=0)   # +3 channels

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

#############################
#  NiftiDataset Class
#############################

class NiftiDataset(BaseDataset):
    """
    A dataset class for volumes with annotations loaded from NIFTI files.
    This class replicates the original pipeline but uses VolumeContainerNIFTI.
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.volnums = [] if opt.trainvols == '-1' else sorted([int(x) for x in opt.trainvols.split(',')])
        self.valnum = opt.validationvol
        self.max_gt_dist = max([int(x) for x in opt.gt_distances.split(',')])

        self.volumes = {}
        for volnum in self.volnums:
            self.volumes[volnum] = VolumeContainerNIFTI(opt, volnum)

        self.valvol = VolumeContainerNIFTI(opt, self.valnum)
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
            tot_samples += int(np.sum(list(self.volumes[volnum].sint_seg_lens.values())))
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


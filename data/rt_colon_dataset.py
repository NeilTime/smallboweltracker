import os.path
from data.base_dataset import BaseDataset
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import pydicom


class VolumeContainer:
    """A container class for a volume with all its small intestine points
    """

    def __init__(self, opt, imnum, inference=False):
        print('Initialising volume: {}'.format(imnum))
        self.opt = opt
        self.set_root = opt.dataroot + opt.masterpath.replace('##', f'{imnum:02d}')
        main_load_folder = self.set_root + '/MR80/'

        im3d, ps, im_or, im_pos, im_end_pos = load_3d(main_load_folder)

        self.data = torch.Tensor(im3d).unsqueeze(0)
        if opt.normalisation != 'median':
            print('only median normalisation supported for this loader')
            quit(1)

        data_median = torch.median(self.data[self.data>10])
        self.data = self.data / data_median
        self.data = torch.clamp(self.data, 0, self.opt.clip_norm)

        nclass = opt.nclass
        self.vertices = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + f'/../vertices{nclass}.txt')
        self.vw = opt.isosample_spacing
        self.pw = opt.patch_size
        self.interp = opt.interp

        self.shape = self.data.shape
        self.spacing = ps

        self.sint_seg_lens = {}
        self.sint_seg_totlen = 0
        gt_dists = [int(x) for x in self.opt.gt_distances.split(',')]

        if not inference:
            self.sint_segs_dense_vox = {}
            trace_files = []
            for fname in os.listdir(self.set_root):
                if 'midline' in fname:
                    fullfname = self.set_root + '/' + fname
                    trace_files.append(fullfname)

            for i in range(len(trace_files)):
                key = str(i + 1)
                TRACE_FILE = sorted(trace_files)[0].replace('_1_midline',f'_{key}_midline').replace('_10_midline',f'_{key}_midline')
                print(TRACE_FILE)

                raw_cl = np.genfromtxt(TRACE_FILE, delimiter=',')
                cl = raw_cl - [1, 1, 0]

                self.sint_segs_dense_vox[key] = cl

            # count the number of gt points in this volume
            for seg in sorted(self.sint_segs_dense_vox.keys()):
                self.sint_seg_lens[seg] = max(0,self.sint_segs_dense_vox[seg].shape[0] - max(gt_dists) * 2)
                self.sint_seg_totlen += self.sint_seg_lens[seg]

            # no segmentations in RT colon set, keep ref for convenience
            self.seggt = torch.zeros(self.data.shape[1:])

    def get_patch(self, loc, rotate=True, with_seg=False, loc_patchspace=False, convert_to_mm=False):
        """Draw a raw patch from the volume, to be used in tracker inference"""
        spacing = self.spacing
        pw = self.pw
        vw = self.vw

        if loc_patchspace:
            #loc = vox_coords_to_mm(loc, spacing)
            loc = [loc[i] * vw for i in range(len(loc))]
            #print(f'#debug new loc {loc}')
        if convert_to_mm:
            loc = vox_coords_to_mm(loc, spacing)

        locx, locy, locz = loc

        rot_matrix, inv_matrix = get_multiax_rotation_matrix(rotate, iters=3)
        im_patch = draw_sample_4D_world_fast(self.data, locx, locy, locz, spacing,
                                             np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                             rot_matrix, interpolation=self.interp).float()
        im_patch = im_patch.clamp(0, self.opt.clip_norm)

        if with_seg:
            seggt_patch = draw_sample_3D_world_fast(self.seggt, locx, locy, locz, spacing,
                                                    np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                                    rot_matrix, interpolation='nearest').float()

            return im_patch, seggt_patch, rot_matrix, inv_matrix
        else:
            return im_patch, rot_matrix, inv_matrix

    def get_sample(self, intest_num, intest_index, valvol=False):
        """
        Adapted from Jelmer's code:
        https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
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

        # Intestine marker coordinates
        intest_coords_vox = self.sint_segs_dense_vox[str(intest_num)][:, :]
        intest_coords_mm = vox_coords_to_mm(intest_coords_vox, spacing)

        locx = intest_coords_mm[intest_index, 0]
        locy = intest_coords_mm[intest_index, 1]
        locz = intest_coords_mm[intest_index, 2]
        point_orig = np.reshape(np.array([locx, locy, locz]), (1, 3))

        # Apply random displacement to location (only in training)
        if self.opt.isTrain:
            # generate a random normal vector
            diff_coord = intest_coords_mm[intest_index - 1, :] - intest_coords_mm[intest_index, :]
            normvec = np.cross(diff_coord, np.random.random((1, 3)))[0]
            normvec = normvec / (np.linalg.norm(normvec) + 1e-6)
            locx = locx + np.random.normal(0, displace_sigma, 1) * normvec[0]
            locy = locy + np.random.normal(0, displace_sigma, 1) * normvec[1]
            locz = locz + np.random.normal(0, displace_sigma, 1) * normvec[2]

        # Compute displacement between next point and current point
        if self.opt.only_displace_input:
            point = point_orig
        else:
            point = np.reshape(np.array([locx, locy, locz]), (1, 3))

        # Get rotation matrices (identity if rotate is False)
        # NOTE! Rotation matrix in draw_sample_3D_world is applied to _coordinates_.
        # By rotating the sampling coordinates, the patch effectively gets the inverse transformation.
        rot_matrix, inv_matrix = get_multiax_rotation_matrix(rotate, iters=3)

        im_patch = draw_sample_4D_world_fast(self.data, locx, locy, locz, spacing,
                                             np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                             rot_matrix, interpolation=self.interp).float()

        seggt_patch = draw_sample_3D_world_fast(self.seggt, locx, locy, locz, spacing,
                                             np.array([pw, pw, pw]), np.array([vw, vw, vw]),
                                             rot_matrix, interpolation='nearest').float()

        target_list = []
        for gt_dist in gt_dists:
            nextp = intest_coords_mm[intest_index - gt_dist, :]
            prevp = intest_coords_mm[intest_index + gt_dist, :]

            displacement = (nextp - point)
            displacement = gt_spacing * gt_dist * displacement / (np.linalg.norm(displacement) + 1e-6)
            displacement_back = (prevp - point)
            displacement_back = gt_spacing * gt_dist * displacement_back / (np.linalg.norm(displacement_back)+1e-6)

            # forward and backwards
            target = self.directionToClass(vertices, displacement, rotMatrix=inv_matrix, sigma=gt_sigma)
            target += self.directionToClass(vertices, displacement_back, rotMatrix=inv_matrix, sigma=gt_sigma)
            target = (target / np.sum(target))
            target_list.append(torch.as_tensor(target).float())

        target = torch.stack(target_list, dim=0)
        im_patch = im_patch.clamp(0, self.opt.clip_norm)
        return im_patch, target, seggt_patch

    def directionToClass(self, vertices, target, rotMatrix=np.eye(4, dtype='float32'), sigma=0):
        """
        Adapted Jelmer's code from
        https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
        Assigns values based on distance to the gt (original spiking version for sigma=0)

        """
        vertexlength = np.linalg.norm(np.squeeze(vertices[0, :]))
        target_orig = target
        target = target.reshape((1, 3))
        target = np.dot(rotMatrix, np.array([target[0, 0], target[0, 1], target[0, 2], 0.0]))
        target = target[:3]
        target = target / (np.linalg.norm(target) / vertexlength)

        dist_to_vert = np.linalg.norm(vertices - target, axis=1)
        distro = np.zeros(dist_to_vert.shape, dtype='float32')

        if sigma == 0:
            # Set label at closest vertice
            distro[np.argmin(dist_to_vert)] = 1.0
        else:
            distro = np.clip(sigma - dist_to_vert, 0, sigma)
        return distro


class RtColonDataset(BaseDataset):
    """
    A dataset class for volume with small intestine annotations, loaded from a hdf5 set.
    Addressed by individual points on the annotation manifold
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.volnums = [] if opt.trainvols == '-1' else sorted([int(x) for x in opt.trainvols.split(',')])
        self.valnum = opt.validationvol
        self.max_gt_dist = max([int(x) for x in self.opt.gt_distances.split(',')])

        self.volumes = {}
        for volnum in self.volnums:
            self.volumes[volnum] = VolumeContainer(opt, volnum)

        self.valvol = VolumeContainer(opt, self.valnum)
        self.valvol.valvol = True
        self.valbatch_nums = []

        np.random.seed(42)  # make sure the same validation batch is sampled every time

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
                        int_index = local_counter + self.max_gt_dist
                        return volnum, int_num, int_index

        print('index {} out of range! Bug in dataset indexing'.format(index))
        quit()

    def __getitem__(self, index):
        """Return a data point

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B
            A (tensor) - - an image in the input domain
            B (tensor) - - The desired activations on the sphere
            C (tensor) - - The ground truth segmentation for this patch
            D (array)  - - Bool array indicating whether centerline GT is available
        """
        volnum, int_num, int_index = self.get_gt_by_index(index)

        # sometimes ignore int_num/int_index in favour of a random non-cl patch during training
        centerline_patch = not (self.opt.isTrain and np.random.rand() < self.opt.non_centerline_ratio)
        if centerline_patch:
            imdata, gtdata, seggtdata = self.volumes[volnum].get_sample(int_num, int_index)
        else:
            gtdata = torch.ones((1,500))*-1
            patchvol = self.volumes[volnum]
            #patchloc = vox_coords_to_mm((np.random.rand() * patchvol.shape[2], np.random.rand() * patchvol.shape[1], np.random.rand() * patchvol.shape[0]), patchvol.spacing)
            patchloc = (np.random.rand() * patchvol.shape[2], np.random.rand() * patchvol.shape[1], np.random.rand() * patchvol.shape[0])
            #print(f'#### Debug, patchloc: {patchloc}')
            imdata, seggtdata, rotmat, irotmat = patchvol.get_patch(patchloc, with_seg=True)
            #print(f'#### Debug, imdata sum: {torch.sum(imdata)}')

        return {'A': imdata, 'B': gtdata, 'C': seggtdata, 'D': centerline_patch}

    def __len__(self):
        """Return the total number of samples in the dataset."""
        tot_samples = 0
        for volnum in self.volnums:
            tot_samples += np.sum(list(self.volumes[volnum].sint_seg_lens.values()))

        return tot_samples

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--normalisation', type=str, default='median',
                            help='normalisation strategy. [median | mean | max | none]')
        parser.add_argument('--masterpath', type=str,
                            default='/MOT3D_multi_tslice_MII##a.hdf5',
                            help='name of the dataset files (## is regexed to 02d)')
        parser.add_argument('--clip_norm', type=int, default=4, help='maximum clamp value after normalisation')
        parser.add_argument('--nclass', type=int, default=500, help='number of classes on the sphere')
        parser.add_argument('--trainvols', type=str, default='4,5,7,10,13,14,15',
                            help='training volume IDs, comma seperated')
        parser.add_argument('--validationvol', type=int, default=18, help='validation volume ID')
        parser.add_argument('--patch_size', type=int, default=19, help='patch size')
        parser.add_argument('--tslice_start', type=int, default=2, help='first tslice (twidth equals input_nc)')
        parser.add_argument('--value_augmentation', type=float, default=0,
                            help='image scaling augmentation factor width (0 for disable)')
        parser.add_argument('--rotation_augmentation', type=bool, default=True, help='rotation augmentations')
        parser.add_argument('--displace_augmentation_mm', type=float, default=1.0, help='displacement aug dist in mm')
        parser.add_argument('--only_displace_input', action='store_true', help='do not recompute gt angle')
        parser.add_argument('--gt_sigma', type=float, default=0, help='sigma of gt smoothing gauss')
        parser.add_argument('--gt_distances', type=str, default='3',
                            help='index distances from center to GT, comma separated')
        parser.add_argument('--isosample_spacing', type=float, default=1.0, help='iso sampling spacing in mm')
        parser.add_argument('--orig_gt_spacing', type=float, default=0.5, help='resolution of ground truth points (mm)')
        parser.add_argument('--seg_bce_factor', type=float, default=1, help='factor of seg BCE loss component')
        parser.add_argument('--selfconsistency_factor', type=float, default=0, help='factor selfconsistency loss component')
        parser.add_argument('--selfconsistency_range', type=int, default=4, help='length of selfconsistency rays')
        parser.add_argument('--selfconsistency_delay', type=int, default=16, help='delay selfconsistency in epochs')
        parser.add_argument('--dir_bce_factor', type=float, default=100, help='factor of direction BCE loss component')
        parser.add_argument('--dir_bce_offset', type=float, default=1, help='offset for nicer plotting')
        parser.add_argument('--interp', type=str, default='linear', help='nearest|[linear]')

        return parser


def vox_coords_to_mm(coords, spacing):
    """ Convert voxel coordinates to mm distance from the origin"""
    if np.size(coords) == 3:
        return [coords[i] * spacing[i] for i in range(3)]

    coords_mm = []
    for coord in coords:
        coords_mm.append([coord[i] * spacing[i] for i in range(3)])

    return np.array(coords_mm)


def mm_coords_to_vox(coords, spacing):
    """ Convert mm distance from the origin to voxel coordinates"""
    if np.size(coords) == 3:
        return [coords[i] / spacing[i] for i in range(3)]

    coords_mm = []
    for coord in coords:
        coords_mm.append([coord[i] / spacing[i] for i in range(3)])

    return np.array(coords_mm)


def get_multiax_rotation_matrix(rotate=True, iters=3):
    """ Chain multiple random rotations"""
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
    """
    Jelmer's rotation matrix code from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    Bit of a monster, but it works.
    """
    if not rotate:
        return np.eye(4, dtype='float32'), np.eye(4, dtype='float32')
    else:
        # Constrain angles between 0 and 90 degrees --> 31-07-17 No, to 360
        rotangle = np.random.randint(0, 3, 1)
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        if rotangle == 0:
            alpha = (np.squeeze(np.float(np.random.randint(0, 360, 1))) / 180.) * np.pi
        if rotangle == 1:
            beta = (np.squeeze(np.float(np.random.randint(0, 360, 1))) / 180.) * np.pi
        if rotangle == 2:
            gamma = (np.squeeze(np.float(np.random.randint(0, 360, 1))) / 180.) * np.pi

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        rotMatrix = np.array([[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg, 0.],
                              [cb * sg, ca * cg + sa * sb * sg, -1. * cg * sa + ca * sb * sg, 0.],
                              [-1 * sb, cb * sa, ca * cb, 0],
                              [0, 0, 0, 1.]])
        alpha = -1.0 * alpha
        beta = -1.0 * beta
        gamma = -1.0 * gamma
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        invMatrix = np.array([[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg, 0.],
                              [cb * sg, ca * cg + sa * sb * sg, -1. * cg * sa + ca * sb * sg, 0.],
                              [-1 * sb, cb * sa, ca * cb, 0],
                              [0, 0, 0, 1.]])
        return rotMatrix, invMatrix


def fast_trilinear_4d(input_array, x_indices, y_indices, z_indices):
    """
    Modified Jelmer's code from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    Uses full first dimension (t slicing should happen before input array is supplied)
    TODO?: currently zero-pads _before_ interpolation
    """
    if len(input_array.shape) != 4:
        print('input array to trilinear_4d is not 4D!!')
        1/0

    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # #Check if xyz1 is beyond array boundary:
    x0[np.where(x0 >= input_array.shape[1])] = input_array.shape[1]
    y0[np.where(y0 >= input_array.shape[2])] = input_array.shape[2]
    z0[np.where(z0 >= input_array.shape[3])] = input_array.shape[3]
    x1[np.where(x1 >= input_array.shape[1])] = input_array.shape[1]
    y1[np.where(y1 >= input_array.shape[2])] = input_array.shape[2]
    z1[np.where(z1 >= input_array.shape[3])] = input_array.shape[3]
    x0[np.where(x0 < 0)] = input_array.shape[1]
    y0[np.where(y0 < 0)] = input_array.shape[2]
    z0[np.where(z0 < 0)] = input_array.shape[3]
    x1[np.where(x1 < 0)] = input_array.shape[1]
    y1[np.where(y1 < 0)] = input_array.shape[2]
    z1[np.where(z1 < 0)] = input_array.shape[3]

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

    input_array = F.pad(input_array, (0,1,0,1,0,1,0,0), mode='constant')

    single_t_patches = []
    for t in range(input_array.shape[0]):
        single_t_array = input_array[t, :, :, :]
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


def draw_sample_4D_world_fast(image, x, y, z, imagespacing, patchsize, patchspacing,
                              rot_matrix=np.eye(4, dtype='float32'), interpolation='nearest'):
    """
    Jelmer's code from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    expects coordinates in mm-form
    """
    patchmargin = (patchsize - 1) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    xs = (x + (unra[0] - patchmargin[0]) * patchspacing[0]) / imagespacing[0]
    ys = (y + (unra[1] - patchmargin[1]) * patchspacing[1]) / imagespacing[1]
    zs = (z + (unra[2] - patchmargin[2]) * patchspacing[2]) / imagespacing[2]

    xs = xs - (x / imagespacing[0])
    ys = ys - (y / imagespacing[1])
    zs = zs - (z / imagespacing[2])

    coords = np.concatenate((np.reshape(xs, (1, xs.shape[0])), np.reshape(ys, (1, ys.shape[0])),
                             np.reshape(zs, (1, zs.shape[0])), np.zeros((1, xs.shape[0]), dtype='float32')), axis=0)

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


def fast_trilinear(input_array, x_indices, y_indices, z_indices):
    """
    Modified Jelmer's code from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    TODO: currently zero-pads _before_ interpolation
    """
    x0 = x_indices.astype(np.integer)
    y0 = y_indices.astype(np.integer)
    z0 = z_indices.astype(np.integer)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # #Check if xyz1 is beyond array boundary:
    x0[np.where(x0 >= input_array.shape[0])] = input_array.shape[0]
    y0[np.where(y0 >= input_array.shape[1])] = input_array.shape[1]
    z0[np.where(z0 >= input_array.shape[2])] = input_array.shape[2]
    x1[np.where(x1 >= input_array.shape[0])] = input_array.shape[0]
    y1[np.where(y1 >= input_array.shape[1])] = input_array.shape[1]
    z1[np.where(z1 >= input_array.shape[2])] = input_array.shape[2]
    x0[np.where(x0 < 0)] = input_array.shape[0]
    y0[np.where(y0 < 0)] = input_array.shape[1]
    z0[np.where(z0 < 0)] = input_array.shape[2]
    x1[np.where(x1 < 0)] = input_array.shape[0]
    y1[np.where(y1 < 0)] = input_array.shape[1]
    z1[np.where(z1 < 0)] = input_array.shape[2]

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

    input_array = F.pad(input_array, (0,1,0,1,0,1), mode='constant')
    output = (input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z) + input_array[x1, y0, z0] * x * (1 - y) * (1 - z) +
              input_array[x0, y1, z0] * (1 - x) * y * (1 - z) + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z +
              input_array[x1, y0, z1] * x * (1 - y) * z + input_array[x0, y1, z1] * (1 - x) * y * z + input_array[
                  x1, y1, z0] * x * y * (1 - z) + input_array[x1, y1, z1] * x * y * z)
    return output


def fast_nearest(input_array, x_indices, y_indices, z_indices):
    """
    Modified Jelmer's code from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    This version zero-pads
    """
    x_ind = (x_indices + 0.5).astype(np.integer)
    y_ind = (y_indices + 0.5).astype(np.integer)
    z_ind = (z_indices + 0.5).astype(np.integer)

    x_ind[np.where(x_ind >= input_array.shape[0])] = input_array.shape[0]
    y_ind[np.where(y_ind >= input_array.shape[1])] = input_array.shape[1]
    z_ind[np.where(z_ind >= input_array.shape[2])] = input_array.shape[2]
    x_ind[np.where(x_ind < 0)] = input_array.shape[0]
    y_ind[np.where(y_ind < 0)] = input_array.shape[1]
    z_ind[np.where(z_ind < 0)] = input_array.shape[2]
    input_array = F.pad(input_array, (0,1,0,1,0,1), mode='constant')
    return input_array[x_ind, y_ind, z_ind]


def draw_sample_3D_world_fast(image, x, y, z, imagespacing, patchsize, patchspacing,
                              rot_matrix=np.eye(4, dtype='float32'), interpolation='nearest'):
    """
    Jelmer's code from
    https://gitlab.amc.nl/qia/jelmer/coronary-centerline-extraction/blob/master/TrainCenterlineExtractor.py
    """
    patchmargin = (patchsize - 1) / 2
    unra = np.unravel_index(np.arange(np.prod(patchsize)), patchsize)
    xs = (x + (unra[0] - patchmargin[0]) * patchspacing[0]) / imagespacing[0]
    ys = (y + (unra[1] - patchmargin[1]) * patchspacing[1]) / imagespacing[1]
    zs = (z + (unra[2] - patchmargin[2]) * patchspacing[2]) / imagespacing[2]

    xs = xs - (x / imagespacing[0])
    ys = ys - (y / imagespacing[1])
    zs = zs - (z / imagespacing[2])

    coords = np.concatenate((np.reshape(xs, (1, xs.shape[0])), np.reshape(ys, (1, ys.shape[0])),
                             np.reshape(zs, (1, zs.shape[0])), np.zeros((1, xs.shape[0]), dtype='float32')), axis=0)

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


def load_3d(scan_path):
    files = []
    for fname in os.listdir(scan_path):
        if 'mat' not in fname:
            fullfname = scan_path + '/' + fname
            files.append(pydicom.read_file(fullfname))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SpacingBetweenSlices
    ps.append(ss)
    tpoints = slices[0].NumberOfTemporalPositions
    im_or = slices[0].ImageOrientationPatient
    im_pos = slices[0].ImagePositionPatient
    im_end_pos = slices[-1].ImagePositionPatient

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d.T

    return img3d, ps, im_or, im_pos, im_end_pos

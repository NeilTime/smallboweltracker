# run_dummy.py
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import dominate
print(dominate.__version__)
import numpy as np
from matplotlib.widgets import Slider


# Adjust the import below according to your project structure.
# For example, if your dataset file is located in the "data" folder as louis_dataset.py:
from data.louis_dataset import louisDataset

# Create a dummy options object with the necessary attributes.
class DummyOpt:
    dataroot = r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility"  # Change to your data root
    masterpath = "MOT3D_multi_tslice_MII##a.hdf5"
    seg_path = "masked_segmentations/MII##a_seg_t2_5c.nii"  # Change to your segmentation folder
    batch_size = 1
    input_nc = 1
    normalisation = "median"
    clip_norm = 4
    gt_distances = "3"
    nclass = 500
    isosample_spacing = 1.0
    patch_size = 32
    interp = "linear"
    orig_gt_spacing = 0.5
    displace_augmentation_mm = 1.0
    gt_sigma = 0
    rotation_augmentation = True
    only_displace_input = False
    isTrain = True
    non_centerline_ratio = 0.2
    # Add the missing attributes:
    trainvols = "4,5,7,10,13,14,15"
    validationvol = 18
    independent_dir = False

opt = DummyOpt()

# Initialize the dataset and a DataLoader for convenience.
dataset = louisDataset(opt)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

while True:
    sample = next(iter(dataloader))
    print(sample['A'].shape, sample['C'].shape)
    print(sample['B'])
    print(sample['D'])
    im_patch = sample['A'][0, 0].numpy()  # shape: (patch_size, patch_size, patch_size) or similar
    seg_patch = sample['C'][0, 0].numpy() # shape: (patch_size, patch_size, patch_size) or similar

    print("Segmentation patch shape:", seg_patch.shape)

    # For visualization, we can choose a slice from the 3D patch.
    # Here, we take the middle slice of the patch.
    slice_idx = im_patch.shape[0] // 2

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(im_patch[slice_idx, :, :], cmap="gray")
    plt.title("Image Patch (middle slice)")

    plt.subplot(1, 2, 2)
    plt.imshow(seg_patch[slice_idx, :, :], cmap="jet")
    plt.title("Segmentation Patch (middle slice)")
    plt.show()

    
    # # ---------------------------
    # # Set up interactive visualization
    # # ---------------------------
    # num_slices = im_patch.shape[2]  # number of slices along z

    # fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    # plt.subplots_adjust(bottom=0.25)

    # # Get a discrete colormap (for example, tab10 provides 10 distinct colors)
    # cmap = plt.get_cmap('tab10')

    # def update_slice(slice_idx):
    #     # Extract the MRI slice (transposed for correct orientation)
    #     mri_slice = im_patch[:, :, slice_idx].T
    #     seg_slice = seg_patch[:, :, slice_idx]
    #     cl_slice = centerline_mask[:, :, slice_idx].T  # 2D mask for current slice

    #     # Clear axes
    #     ax_left.clear()
    #     ax_right.clear()

    #     # Display the MRI slice
    #     ax_left.imshow(mri_slice, cmap='gray')
    #     ax_left.set_title(f"MRI Slice {slice_idx} with Centerline Points")

    #     # For each unique label in this slice, plot its points as dots
    #     for lab in np.unique(cl_slice[cl_slice != 0]):
    #         rows, cols = np.where(cl_slice == lab)
    #         if rows.size > 0:
    #             # Use a discrete colormap to pick a color per label
    #             color = cmap(int(lab) % cmap.N)
    #             # Plot the points as dots using marker '.'
    #             ax_left.scatter(cols, rows, s=20, color=color, marker='.', label=f"Label {lab}")

    #     # Add a legend if any centerline points were plotted
    #     if np.any(cl_slice != 0):
    #         ax_left.legend(loc='upper right')

    #     # Display the segmentation slice
    #     ax_right.imshow(seg_slice, cmap='jet')
    #     ax_right.set_title(f"Segmentation Slice {slice_idx}")

    #     fig.canvas.draw_idle()


    # # Initialize with the middle slice
    # initial_slice = num_slices // 2
    # update_slice(initial_slice)

    # # Create a slider to scroll through slices
    # ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    # slice_slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=initial_slice, valfmt='%0.0f')
    # slice_slider.on_changed(lambda val: update_slice(int(val)))

    # plt.show()

    # # Ask the user if they want to continue
    # cont = input("Continue? [y/n]: ")
    # if cont.lower() != "y":
    #     break
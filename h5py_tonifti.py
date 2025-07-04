import os
import h5py
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import binary_dilation

# ---------------------------
# Load MRI and centerline mask from HDF5
# ---------------------------
mri_file_path = r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility\MOT3D_multi_tslice_MII04a.hdf5"

with h5py.File(mri_file_path, "r") as f:
    # Load the MRI scan; assume shape is (T, X, Y, Z)
    dset = f["MOT3DBH"]                       # keep the Dataset object
    annotation_t_slice = dset.attrs["annotation_tslice"]  # ← attribute
    mri_scan = dset[:]                        # now pull the pixel data
    # Attributes needed for NIfTI header
    spacing = dset.attrs["spacing"]  # [x, y, z] voxel spacing
    patient_position = dset.attrs["patient_position"]  # 3D origin (in mm)
    patient_orientation = dset.attrs["patient_orientation"]  # 6-element orientation vector
    tslices = dset.attrs["tslices"]  # Number of time points
    
    # Create an empty mask for centerlines
    centerline_mask = np.zeros_like(mri_scan, dtype=np.uint8)
    
    centerline_groups = ["sint_segs_dense_vox"]
    label = 1
    for group_name in centerline_groups:
        if group_name in f:
            group = f[group_name]
            for dataset_name in group.keys():
                centerline = group[dataset_name][:]  # centerline points (N, 3)
                voxel_indices = centerline.astype(int)
                for x, y, z in voxel_indices:
                    centerline_mask[x - 1, y - 1, z] = label
                label += 1

# ---------------------------
# Dilate the centerline mask (slice-by-slice)
# ---------------------------
# Increase the structuring element size to make centerlines more visible
structure = np.ones((5, 5), dtype=bool)
dilated_mask = np.zeros_like(centerline_mask)
for z in range(centerline_mask.shape[2]):
    labels_in_slice = np.unique(centerline_mask[:, :, z])
    labels_in_slice = labels_in_slice[labels_in_slice != 0]
    for lab in labels_in_slice:
        binary = (centerline_mask[:, :, z] == lab)
        dilated = binary_dilation(binary, structure=structure)
        dilated_mask[:, :, z][dilated] = lab

# ---------------------------
# Load segmentation from NIfTI file
# ---------------------------
seg_nii_path = r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility\masked_segmentations\MII04a_seg_t2_5c.nii"
seg_img = nib.load(seg_nii_path)
seg_data = seg_img.get_fdata().astype(np.uint8)
# If necessary, transpose seg_data; here adjust if your segmentation is already aligned:
seg_data = np.transpose(seg_data, (1, 2, 0))
print(np.unique(seg_data))
seg_data = seg_data == 1
print(seg_data.shape)

# ---------------------------
# Set up interactive visualization
# ---------------------------
num_slices = mri_scan.shape[2]  # number of slices along z

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Get a discrete colormap (for example, tab10 provides 10 distinct colors)
cmap = plt.get_cmap('tab10')

def update_slice(slice_idx):
    # Extract the MRI slice (transposed for correct orientation)
    mri_slice = mri_scan[:, :, slice_idx].T
    seg_slice = seg_data[:, :, slice_idx]
    cl_slice = centerline_mask[:, :, slice_idx].T  # 2D mask for current slice

    # Clear axes
    ax_left.clear()
    ax_right.clear()

    # Display the MRI slice
    ax_left.imshow(mri_slice, cmap='gray')
    ax_left.set_title(f"MRI Slice {slice_idx} with Centerline Points")

    # For each unique label in this slice, plot its points as dots
    for lab in np.unique(cl_slice[cl_slice != 0]):
        rows, cols = np.where(cl_slice == lab)
        if rows.size > 0:
            # Use a discrete colormap to pick a color per label
            color = cmap(int(lab) % cmap.N)
            # Plot the points as dots using marker '.'
            ax_left.scatter(cols, rows, s=20, color=color, marker='.', label=f"Label {lab}")

    # Add a legend if any centerline points were plotted
    if np.any(cl_slice != 0):
        ax_left.legend(loc='upper right')

    # Display the segmentation slice
    ax_right.imshow(seg_slice, cmap='jet')
    ax_right.set_title(f"Segmentation Slice {slice_idx}")

    fig.canvas.draw_idle()


# Initialize with the middle slice
initial_slice = num_slices // 2
update_slice(initial_slice)

# Create a slider to scroll through slices
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
slice_slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=initial_slice, valfmt='%0.0f')
slice_slider.on_changed(lambda val: update_slice(int(val)))

plt.show()

# ---------------------------
# Prepare affine matrix
# ---------------------------
# Build affine from spacing, orientation and position
# Interpretation of orientation vector from the HDF5 file:
# Assuming first 3 elements are x-direction cosine, next 3 are y-direction cosine
# Third direction (z-direction) is cross product of first two.

orientation_x = np.array(patient_orientation[:3])
orientation_y = np.array(patient_orientation[3:6])
orientation_z = np.cross(orientation_x, orientation_y)

affine = np.eye(4)
affine[:3, 0] = orientation_x * spacing[0]
affine[:3, 1] = orientation_y * spacing[1]
affine[:3, 2] = orientation_z * spacing[2]
affine[:3, 3] = patient_position  # Image origin (in mm)

print("Affine matrix for NIfTI:")
print(affine)

# ---------------------------
# Save as NIfTI
# ---------------------------
# Save each time point as a separate 3D volume inside a 4D NIfTI
output_nii_path = r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility\MII04a_mri.nii"

# Ensure MRI scan shape is in (X, Y, Z, T) order for nibabel
mri_scan_nifti_order = np.transpose(mri_scan, (1, 2, 3, 0))  # From (T, X, Y, Z) → (X, Y, Z, T)

# Create NIfTI image
nii_img = nib.Nifti1Image(mri_scan_nifti_order, affine)

# Optionally: Fill header with scale info
nii_img.header['pixdim'][1:4] = spacing

# Save
nib.save(nii_img, output_nii_path)

print(f"MRI saved to: {output_nii_path}")
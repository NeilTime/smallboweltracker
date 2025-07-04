import h5py
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt 

# Load the file
file_path = "data_motility/MOT3D_multi_tslice_MII05a.hdf5"
file_path_nifti = "data_motility/nii_t2.nii"

with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")
            # Print attributes
            for attr in obj.attrs:
                print(f"    Attribute: {attr} = {obj.attrs[attr]}")

    f.visititems(print_structure)


with h5py.File(file_path, "r") as f:
    # Extract the MRI scan dataset
    dset = f["MOT3DBH"]                       # keep the Dataset object
    annotation_t_slice = dset.attrs["annotation_tslice"]  # ← attribute
    mri_scan = dset[:]                        # now pull the pixel data

# Plot the first and middle slices side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

mid_idx = mri_scan.shape[3] // 2
# First slice
axes[0].imshow(mri_scan[annotation_t_slice, :, :, mid_idx], cmap="gray")
axes[0].set_title("First Slice")

# Middle slice
axes[1].imshow(mri_scan[10, :, :, mid_idx], cmap="gray")
axes[1].set_title("Middle Slice")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

with h5py.File(file_path, "r") as f:
    # Extract first segmentation centerline
    centerline = f["lint_segs_dense_vox/1"][:]

# Print first 5 (x, y, z) coordinates
print(centerline[:5])

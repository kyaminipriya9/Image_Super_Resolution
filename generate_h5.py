import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# -----------------------------
# PATHS (LOCAL)
# -----------------------------
IMAGE_DIR = "SRCNN/images"          # input images
SAVE_PATH = "SRCNN/data/train_mscale.h5"

# -----------------------------
# PARAMETERS (SRCNN standard)
# -----------------------------
PATCH_SIZE = 33
SCALE = 3
STRIDE = 14

# -----------------------------
# PATCH EXTRACTION FUNCTION
# -----------------------------
def extract_patches(lr, hr):
    lr_patches, hr_patches = [], []
    w, h = lr.size

    for x in range(0, w - PATCH_SIZE + 1, STRIDE):
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            lr_patch = lr.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            hr_patch = hr.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))

            lr_patches.append(np.array(lr_patch))
            hr_patches.append(np.array(hr_patch))

    return lr_patches, hr_patches

# -----------------------------
# MAIN
# -----------------------------
lr_data, hr_data = [], []

for img_name in tqdm(os.listdir(IMAGE_DIR)):
    img_path = os.path.join(IMAGE_DIR, img_name)

    img = Image.open(img_path).convert("YCbCr")
    y, _, _ = img.split()

    # HR image
    hr = y

    # Create LR image
    lr = hr.resize(
        (hr.width // SCALE, hr.height // SCALE),
        Image.BICUBIC
    )
    lr = lr.resize(
        (hr.width, hr.height),
        Image.BICUBIC
    )

    lr_p, hr_p = extract_patches(lr, hr)
    lr_data.extend(lr_p)
    hr_data.extend(hr_p)

# Convert to numpy
lr_data = np.array(lr_data, dtype=np.float32) / 255.0
hr_data = np.array(hr_data, dtype=np.float32) / 255.0

# Add channel dimension
lr_data = lr_data[:, None, :, :]
hr_data = hr_data[:, None, :, :]

print("LR shape:", lr_data.shape)
print("HR shape:", hr_data.shape)

# Save HDF5
with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("data", data=lr_data)
    f.create_dataset("label", data=hr_data)

print("train_mscale.h5 created successfully")

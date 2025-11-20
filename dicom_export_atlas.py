import os
import sys
import math
import struct
from pathlib import Path

import pydicom
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

FIXED_HU_SHIFT = 1100
MAX_ATLAS_DIM = 256  # max width/height of atlas

def load_dicom(path):
    path = Path(path)
    slices = []
    if path.is_file():
        ds = pydicom.dcmread(str(path))
        data = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        data = data * slope + intercept + FIXED_HU_SHIFT
        # make 3D if single-frame
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        return data
    elif path.is_dir():
        files = sorted(path.glob("*.dcm"))
        if not files:
            files = sorted([f for f in path.iterdir() if f.is_file()])
        for f in files:
            try:
                ds = pydicom.dcmread(str(f))
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                arr = arr * slope + intercept + FIXED_HU_SHIFT
                slices.append((getattr(ds, 'ImagePositionPatient', [0,0,0])[2], arr))
            except Exception:
                continue
        if not slices:
            raise ValueError("No valid DICOM slices found")
        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices])
        return volume
    else:
        raise ValueError("Invalid path")

def reorient_volume(volume):
    # Swap axes Y,X -> X,Y and flip Y for correct display
    volume = np.transpose(volume, (1,0,2))
    volume = np.flip(volume, axis=1)
    volume = np.flip(volume, axis=0)
    return volume

def downscale_volume(volume, max_dim=MAX_ATLAS_DIM):
    z, y, x = volume.shape
    max_current_dim = max(z, y, x)
    if max_current_dim <= max_dim:
        return volume  # no scaling needed

    scale = max_dim / max_current_dim
    volume = zoom(volume, (scale, scale, scale), order=1)
    return volume

def create_atlas(volume):
    depth, height, width = volume.shape
    cols = math.ceil(math.sqrt(depth))
    rows = math.ceil(depth / cols)
    atlas = np.zeros((rows*height, cols*width), dtype=volume.dtype)
    for i in range(depth):
        row = i // cols
        col = i % cols
        atlas[row*height:(row+1)*height, col*width:(col+1)*width] = volume[i]
    return atlas

def save_png(volume, out_path):
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume_norm = ((volume - vmin)/(vmax - vmin)*255).astype(np.uint8)
    else:
        volume_norm = np.zeros_like(volume, dtype=np.uint8)
    atlas = create_atlas(volume_norm)
    Image.fromarray(atlas).save(out_path)
    print(f"PNG atlas saved: {out_path}")

def save_raw(volume, out_path):
    depth, height, width = volume.shape
    volume_f16 = volume.astype(np.float16)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('IIII', width, height, depth, 1))  # format code 1 = float16
        f.write(volume_f16.tobytes())
    print(f"Raw volume saved: {out_path}")

def process(path):
    path = Path(path)
    print(f"\nProcessing: {path}")
    volume = load_dicom(path)
    volume = reorient_volume(volume)
    volume = downscale_volume(volume)
    out_dir = path.parent if path.is_file() else path
    base_name = path.stem if path.is_file() else path.name
    save_png(volume, out_dir / f"{base_name}_atlas.png")
    save_raw(volume, out_dir / f"{base_name}_volume.raw")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            process(p)
    else:
        p = input("Enter path to DICOM file or folder: ").strip('"').strip("'")
        process(p)

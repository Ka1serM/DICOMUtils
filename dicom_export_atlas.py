import pydicom
import numpy as np
from pathlib import Path
import sys
import os
import math
import struct
from PIL import Image

def load_dicom_series(file_path):
    path = Path(file_path)

    if path.is_file():
        ds = pydicom.dcmread(str(path))
        if hasattr(ds, 'pixel_array'):
            volume = ds.pixel_array
            if volume.ndim == 2:
                volume = volume[np.newaxis, :, :]
            return volume.astype(np.float32), path.parent, path.stem
        else:
            raise ValueError("DICOM file has no pixel data")

    elif path.is_dir():
        dicom_files = sorted(path.glob('*.dcm'))
        if not dicom_files:
            dicom_files = sorted([f for f in path.iterdir() if f.is_file()])
        if not dicom_files:
            raise ValueError("No DICOM files found in directory")

        slices = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(str(f))
                if hasattr(ds, 'pixel_array'):
                    z = ds.ImagePositionPatient[2] if hasattr(ds, 'ImagePositionPatient') else 0
                    slices.append((z, ds.pixel_array))
            except Exception:
                continue

        if not slices:
            raise ValueError("No valid DICOM slices with pixel data found")

        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices]).astype(np.float32)
        return volume, path, path.name

    else:
        raise ValueError("Path must be a DICOM file or directory")

def create_atlas(volume):
    """Create 2D atlas from 3D volume for PNG preview."""
    depth, height, width = volume.shape
    cols = math.ceil(math.sqrt(depth))
    rows = math.ceil(depth / cols)

    atlas = np.zeros((rows * height, cols * width), dtype=volume.dtype)
    for i in range(depth):
        row = i // cols
        col = i % cols
        atlas[row*height:(row+1)*height, col*width:(col+1)*width] = volume[i]
    return atlas

def save_png(volume, output_path):
    """Save 2D PNG atlas (normalized)."""
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume_norm = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        volume_norm = np.zeros_like(volume, dtype=np.uint8)

    atlas = create_atlas(volume_norm)
    img = Image.fromarray(atlas, mode='L')
    img.save(output_path)
    print(f"✓ PNG saved: {output_path} ({atlas.shape[1]}x{atlas.shape[0]})")

def save_raw_volume(volume, output_path):
    """Save 3D volume as raw Float32 for raymarching with header."""
    depth, height, width = volume.shape
    volume_f32 = volume.astype(np.float32)  # Convert to 32-bit float

    with open(output_path, 'wb') as f:
        # Header: width, height, depth, format=2 (Float32)
        f.write(struct.pack('IIII', width, height, depth, 2))
        # Flatten depth-major: slice0, slice1, ...
        f.write(volume_f32.tobytes())

    print(f"✓ Raw Float32 volume saved: {output_path} ({width}x{height}x{depth})")

def process_dicom(file_path):
    try:
        print(f"\nProcessing: {file_path}")
        volume, out_dir, base_name = load_dicom_series(file_path)
        print(f"Loaded volume shape: {volume.shape}")

        # Paths
        png_path = out_dir / f"{base_name}_atlas.png"
        raw_path = out_dir / f"{base_name}_volume.raw"

        # Save both
        save_png(volume, png_path)
        save_raw_volume(volume, raw_path)

        print(f"✓ Done: {file_path}")
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    print("="*60)
    print("DICOM to PNG + Raw Float16 Volume Exporter")
    print("="*60)

    if len(sys.argv) > 1:
        for file_path in sys.argv[1:]:
            file_path = file_path.strip('"').strip("'")
            if os.path.exists(file_path):
                process_dicom(file_path)
            else:
                print(f"✗ File not found: {file_path}")
    else:
        file_path = input("Enter path to DICOM file or folder: ").strip('"').strip("'")
        if os.path.exists(file_path):
            process_dicom(file_path)
        else:
            print(f"✗ File not found: {file_path}")

    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()

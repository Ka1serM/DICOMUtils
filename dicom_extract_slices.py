import pydicom
import numpy as np
from pathlib import Path
import sys
import os
from time import sleep

def load_dicom_series(file_path):
    """Load DICOM series from a directory or a single multi-slice DICOM file."""
    path = Path(file_path)
    slices = []

    print(f"📁 Loading File Path: {file_path}")
    sleep(0.3)

    if path.is_dir():
        print(f"🗂️ Found directory. Searching for DICOM files...")
        dicom_files = sorted(path.glob('*.dcm'))
        if not dicom_files:
            raise ValueError("No DICOM files found in the directory.")
        print(f"🔎 Found {len(dicom_files)} DICOM file(s). Loading slices...")
        for i, f in enumerate(dicom_files, 1):
            print(f"   Loading slice {i}/{len(dicom_files)}: {f.name}", end='\r')
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, 'pixel_array'):
                z = float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else 0
                slices.append((z, ds))
        print("\n✅ All slices loaded.")
        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1].pixel_array for s in slices])
        dicom_ds_list = [s[1] for s in slices]
        output_stem = path.stem

    elif path.is_file():
        ds = pydicom.dcmread(str(path), force=True)
        if not hasattr(ds, 'pixel_array'):
            raise ValueError("The DICOM file has no image data.")
        pixel_array = ds.pixel_array
        if pixel_array.ndim == 2:
            volume = pixel_array[np.newaxis, :, :]
        else:
            volume = pixel_array
        dicom_ds_list = [ds] * volume.shape[0]
        output_stem = path.stem
        path = path.parent
        print(f"✅ DICOM file loaded. Shape: {volume.shape}")

    else:
        raise ValueError("❌ Invalid path. Please provide a folder or a DICOM file.")

    return volume, dicom_ds_list, path, output_stem

def save_slice_as_dicom(slice_array, reference_ds, output_path, axis_name, stem):
    """Save a single slice as a DICOM file using reference metadata."""
    ds = reference_ds.copy()
    ds.PixelData = slice_array.tobytes()

    if slice_array.ndim == 2:
        ds.Rows, ds.Columns = slice_array.shape
    else:
        ds.Rows, ds.Columns = slice_array.shape[1], slice_array.shape[2]

    from pydicom.uid import generate_uid
    ds.SOPInstanceUID = generate_uid()

    filename = output_path / f"{stem}_{axis_name}.dcm"
    ds.save_as(str(filename))
    print(f"💾 Saved {axis_name} slice: {filename}")

def extract_middle_slices(file_path):
    """Extract middle slice along each axis and save as DICOM."""
    volume, dicom_ds_list, path, stem = load_dicom_series(file_path)
    depth, height, width = volume.shape

    axial_idx = depth // 2
    save_slice_as_dicom(volume[axial_idx, :, :], dicom_ds_list[axial_idx], path, "axial", stem)

    sagittal_idx = width // 2
    save_slice_as_dicom(volume[:, :, sagittal_idx], dicom_ds_list[0], path, "sagittal", stem)

    coronal_idx = height // 2
    save_slice_as_dicom(volume[:, coronal_idx, :], dicom_ds_list[0], path, "coronal", stem)

def main():
    print("DICOM Slice Extractor v.0.0.1 by Marcel Kazemi")
    if len(sys.argv) >= 2:
        dicom_path = sys.argv[1]
    else:
        dicom_path = input("➡️ Enter the path to a DICOM folder or file: ").strip()

    if not os.path.exists(dicom_path):
        print(f"❌ Path not found: {dicom_path}")
        sys.exit(1)

    try:
        extract_middle_slices(dicom_path)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

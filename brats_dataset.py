import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os

class BraTSDataset(Dataset):
    def __init__(self, root, split='train'):
        self.root_dir = root
        if split == 'train':
            self.patients = sorted(os.listdir(root))[:1000]
        else:
            self.patients = sorted(os.listdir(root))[1000:]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_dir = os.path.join(self.root_dir, patient_id)

        # Load modalities
        t1     = nib.load(os.path.join(patient_dir, f"{patient_id}-t1n.nii.gz")).get_fdata()
        t1ce   = nib.load(os.path.join(patient_dir, f"{patient_id}-t1c.nii.gz")).get_fdata()
        t2     = nib.load(os.path.join(patient_dir, f"{patient_id}-t2w.nii.gz")).get_fdata()
        flair  = nib.load(os.path.join(patient_dir, f"{patient_id}-t2f.nii.gz")).get_fdata()
        seg    = nib.load(os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")).get_fdata()

        # Stack into multi-channel image
        image = np.stack([t1, t1ce, t2, flair], axis=0)  # shape: (4, H, W, D)

        # Normalize per modality (simple z-score)
        image = (image - image.mean(axis=(1,2,3), keepdims=True)) / (image.std(axis=(1,2,3), keepdims=True) + 1e-8)

        # Convert to torch
        image = torch.from_numpy(image.astype(np.float32))
        seg   = torch.from_numpy(seg.astype(np.int64))  # segmentation is categorical

        return image, seg, patient_id
# 文件: src/datasets.py
# V7.0: 最终修正版 - 适配 VAE 的 [-1, 1] 范围

import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class T1FADataset(Dataset):
    def __init__(self, t1_dir: str, fa_dir: str, preload_ram: bool = False, target_size=(224, 224)):
        self.t1_dir = t1_dir
        self.fa_dir = fa_dir
        self.preload_ram = preload_ram
        self.target_size = target_size

        self.t1_files = sorted(glob.glob(os.path.join(t1_dir, "*.png")))
        if len(self.t1_files) == 0:
            raise FileNotFoundError(f"❌ 错误: 在 {t1_dir} 中没找到 .png 文件！")
            
        self.slice_filenames = [os.path.basename(f) for f in self.t1_files]
        print(f"Dataset initialized. Found {len(self.t1_files)} images. Range: [-1, 1]")

        self.cache = {}
        if self.preload_ram:
            print(f"🚀 Pre-loading images to RAM...")
            for idx in tqdm(range(len(self.t1_files)), desc="Loading"):
                data = self._load_pair(idx)
                if data is not None:
                    self.cache[idx] = data

    def _read_image_safe(self, path):
        try:
            stream = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
            if img is None: return None
            # 转 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 强制 Resize (关键！)
            if self.target_size is not None:
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
            return img
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
            return None

    def _load_pair(self, idx):
        t1_path = self.t1_files[idx]
        filename = self.slice_filenames[idx]
        fa_path = os.path.join(self.fa_dir, filename)

        t1_img = self._read_image_safe(t1_path)
        fa_img = self._read_image_safe(fa_path)

        if t1_img is None or fa_img is None: return None

        # HWC -> CHW
        t1_tensor = torch.from_numpy(t1_img).permute(2, 0, 1).float()
        fa_tensor = torch.from_numpy(fa_img).permute(2, 0, 1).float()

        # 归一化到 [-1, 1] (VAE 标准)
        t1_tensor = (t1_tensor / 127.5) - 1.0
        fa_tensor = (fa_tensor / 127.5) - 1.0

        return {
            "t1_slice": t1_tensor,
            "fa_slice": fa_tensor,
            "fname": filename
        }

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        if self.preload_ram:
            return self.cache.get(idx, None)
        else:
            return self._load_pair(idx)


try:
    from src.data.t1fa_subject_dataset import T1FASubjectSliceDataset
except Exception:
    T1FASubjectSliceDataset = None

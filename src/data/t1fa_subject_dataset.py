import os
import re
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .subject_index import normalize_subject_id


def _read_image(path: str | Path, target_size: tuple[int, int] | None) -> np.ndarray:
    path = str(path)
    stream = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return image


def _parse_slice_id(filename: str) -> int:
    match = re.search(r"_z(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Cannot parse slice id from {filename}")
    return int(match.group(1))


class T1FASubjectSliceDataset(Dataset):
    def __init__(
        self,
        t1_dir: str | Path,
        fa_dir: str | Path,
        subject_index: pd.DataFrame,
        target_size: tuple[int, int] | None = (224, 224),
    ):
        self.t1_dir = Path(t1_dir)
        self.fa_dir = Path(fa_dir)
        self.target_size = target_size
        self.t1_files = sorted(self.t1_dir.glob("*.png"))
        if not self.t1_files:
            raise FileNotFoundError(f"No .png files found in {self.t1_dir}")

        self.subject_metadata: Dict[str, Dict[str, Any]] = {}
        for row in subject_index.to_dict(orient="records"):
            self.subject_metadata[normalize_subject_id(row["subject_id"])] = row

    def __len__(self) -> int:
        return len(self.t1_files)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        image = _read_image(path, self.target_size)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return (tensor / 127.5) - 1.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t1_path = self.t1_files[idx]
        filename = t1_path.name
        fa_path = self.fa_dir / filename
        if not fa_path.exists():
            raise FileNotFoundError(f"Missing paired FA slice: {fa_path}")

        subject_id = normalize_subject_id(filename.split("_z", 1)[0])
        if subject_id not in self.subject_metadata:
            raise KeyError(f"Subject {subject_id} missing from subject index")
        metadata = self.subject_metadata[subject_id]

        sample: Dict[str, Any] = {
            "t1_slice": self._load_tensor(t1_path),
            "fa_slice": self._load_tensor(fa_path),
            "fname": filename,
            "subject_id": subject_id,
            "slice_id": _parse_slice_id(filename),
            "group_id": int(metadata["group_id"]),
            "group_name": str(metadata["group_name"]),
            "age": float(metadata["age"]),
            "gender": int(metadata["gender"]),
            "edu": float(metadata["edu"]),
            "MMSE": float(metadata["MMSE"]),
            "split": str(metadata["split"]),
        }
        return sample


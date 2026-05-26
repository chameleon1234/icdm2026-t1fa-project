import re
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .subject_index import normalize_subject_id


def _parse_slice_id(filename: str) -> int:
    match = re.search(r"_z(\d+)\.png$", filename)
    if not match:
        raise ValueError(f"Cannot parse slice id from {filename}")
    return int(match.group(1))


def _read_gray_tensor(path: Path, target_size: tuple[int, int] | None) -> torch.Tensor:
    stream = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(image).float().unsqueeze(0)
    return (tensor / 127.5) - 1.0


class T1FAStackDataset(Dataset):
    """Return adjacent T1 slices as channels and the center FA slice as target."""

    def __init__(
        self,
        t1_dir: str | Path,
        fa_dir: str | Path,
        context_slices: int = 3,
        target_size: tuple[int, int] | None = (224, 224),
    ):
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError(f"context_slices must be a positive odd integer, got {context_slices}")

        self.t1_dir = Path(t1_dir)
        self.fa_dir = Path(fa_dir)
        self.context_slices = int(context_slices)
        self.target_size = target_size
        self.t1_files = sorted(self.t1_dir.glob("*.png"))
        if not self.t1_files:
            raise FileNotFoundError(f"No .png files found in {self.t1_dir}")

        self._context_paths: Dict[str, List[Path]] = {}
        by_subject: Dict[str, List[Path]] = {}
        for path in self.t1_files:
            subject_id = normalize_subject_id(path.name.split("_z", 1)[0])
            by_subject.setdefault(subject_id, []).append(path)

        half = self.context_slices // 2
        for subject_paths in by_subject.values():
            ordered = sorted(subject_paths, key=lambda item: _parse_slice_id(item.name))
            for center_idx, center_path in enumerate(ordered):
                neighbors: List[Path] = []
                for offset in range(-half, half + 1):
                    neighbor_idx = min(max(center_idx + offset, 0), len(ordered) - 1)
                    neighbors.append(ordered[neighbor_idx])
                self._context_paths[center_path.name] = neighbors

    def __len__(self) -> int:
        return len(self.t1_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        center_path = self.t1_files[idx]
        filename = center_path.name
        fa_path = self.fa_dir / filename
        if not fa_path.exists():
            raise FileNotFoundError(f"Missing paired FA slice: {fa_path}")

        t1_stack = torch.cat(
            [_read_gray_tensor(path, self.target_size) for path in self._context_paths[filename]],
            dim=0,
        )
        fa_slice = _read_gray_tensor(fa_path, self.target_size)
        subject_id = normalize_subject_id(filename.split("_z", 1)[0])
        return {
            "t1_slice": t1_stack,
            "fa_slice": fa_slice,
            "fname": filename,
            "subject_id": subject_id,
            "slice_id": _parse_slice_id(filename),
        }

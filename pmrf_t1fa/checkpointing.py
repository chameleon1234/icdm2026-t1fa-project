import os
from pathlib import Path
from typing import Any, Iterable


def _resume_values_match(key: str, old_value: Any, new_value: Any) -> bool:
    if key.endswith("_ckpt") or key.endswith("_path") or key.endswith("_dir"):
        return os.path.normcase(os.path.normpath(str(old_value))) == os.path.normcase(
            os.path.normpath(str(new_value))
        )
    return old_value == new_value


def select_resume_checkpoint(
    ckpt_dir: str | os.PathLike,
    stage_name: str,
    explicit_path: str | os.PathLike | None = None,
    auto_resume: bool = True,
    prefer_healthy: bool = True,
) -> str | None:
    if explicit_path:
        resume_path = Path(explicit_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        return str(resume_path)
    if not auto_resume:
        return None

    ckpt_root = Path(ckpt_dir)
    candidates = []
    if prefer_healthy:
        candidates.append(ckpt_root / f"healthy_latest_{stage_name}.pt")
    candidates.append(ckpt_root / f"latest_{stage_name}.pt")
    if not prefer_healthy:
        candidates.append(ckpt_root / f"healthy_latest_{stage_name}.pt")

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def validate_resume_args(
    checkpoint: dict[str, Any],
    current_args: Any,
    required_keys: Iterable[str],
    checkpoint_path: str | os.PathLike,
) -> None:
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    mismatches = []
    for key in required_keys:
        if key not in checkpoint_args or not hasattr(current_args, key):
            continue
        old_value = checkpoint_args[key]
        new_value = getattr(current_args, key)
        if not _resume_values_match(key, old_value, new_value):
            mismatches.append(f"{key}: checkpoint={old_value!r}, current={new_value!r}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            f"Cannot resume from incompatible checkpoint {checkpoint_path}. "
            f"Start a new --run_name or match these arguments: {joined}"
        )

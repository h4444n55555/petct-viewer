"""Evaluation-time perturbations for the PETCT-RobustSeg study.

All functions take the *clean, normalised* 2-channel image tensor of shape (2, X, Y, Z)
with channel 0 = PET and channel 1 = CT (both scaled to [0, 1] by data.py), and return a
new tensor of the same shape. The input is never modified.

Design rules (from the project brief)
-------------------------------------
* Perturbations are defined here, BEFORE any evaluation, and their levels are frozen
  (see configs/perturbations_frozen.json and tests/test_week1.py).
* The ground-truth mask is never passed to, nor altered by, any perturbation. Low-resolution
  resampling degrades the image and restores it to the ORIGINAL grid, so the same
  voxel grid (the "resampling map") is used for image, prediction and mask when metrics
  are computed.
* These are synthetic image-quality changes. They are NOT claimed to represent any
  particular scanner or acquisition protocol.
* Randomness is deterministic: seeded from (case_id, perturbation name) via SHA-256,
  never from Python's per-process hash().
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PET, CT = 0, 1
_TARGET_CHANNELS = {"pet": (PET,), "ct": (CT,), "both": (PET, CT)}


# ----------------------------------------------------------------------------- helpers
def _as_float_tensor(x) -> torch.Tensor:
    """Plain float32 torch tensor copy (also accepts numpy arrays and MONAI MetaTensors)."""
    if hasattr(x, "as_tensor"):          # MONAI MetaTensor
        x = x.as_tensor()
    return torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x).float().clone()


def seed_from(case_id: str, name: str) -> int:
    """Deterministic 63-bit seed from a case id and a perturbation name."""
    digest = hashlib.sha256(f"{case_id}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


# ------------------------------------------------------------------- the perturbations
def low_resolution(image, factor: float, channels: Sequence[int] = (PET, CT)) -> torch.Tensor:
    """Degrade spatial resolution by `factor`, then restore the original grid.

    Downsampling uses area (box) averaging, i.e. coarser voxels, with output size
    ceil(size / factor); restoring uses trilinear interpolation to the exact original size.
    factor == 1 returns an exact copy of the input.
    """
    out = _as_float_tensor(image)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return out
    spatial = tuple(out.shape[1:])
    small = tuple(max(1, int(np.ceil(s / factor))) for s in spatial)
    for c in channels:
        vol = out[c][None, None]                                   # (1, 1, X, Y, Z)
        down = F.interpolate(vol, size=small, mode="area")
        up = F.interpolate(down, size=spatial, mode="trilinear", align_corners=False)
        out[c] = up[0, 0]
    return out


def gaussian_noise(image, sigma: float, target: str, case_id: str, name: str | None = None) -> torch.Tensor:
    """Add zero-mean Gaussian noise (std = sigma, in normalised units) to PET, CT or both.

    The result is deliberately NOT clipped back to [0, 1] (clipping would bias the
    background upward). sigma == 0 returns an exact copy of the input.
    """
    if target not in _TARGET_CHANNELS:
        raise ValueError(f"target must be one of {list(_TARGET_CHANNELS)}")
    out = _as_float_tensor(image)
    if sigma == 0:
        return out
    gen = torch.Generator().manual_seed(seed_from(case_id, name or f"noise_{target}_{sigma}"))
    for c in _TARGET_CHANNELS[target]:
        out[c] = out[c] + sigma * torch.randn(out[c].shape, generator=gen)
    return out


def remove_pet(image, background: float = 0.0) -> torch.Tensor:
    """Replace the PET channel with a constant (default 0 = no uptake); CT is untouched."""
    out = _as_float_tensor(image)
    out[PET] = background
    return out


# ------------------------------------------------------------------------ test conditions
@dataclass(frozen=True)
class Spec:
    """One named test condition. `kind` is clean | lowres | noise | pet_removed."""
    name: str
    kind: str
    family: str          # grouping for plots: clean | lowres | noise_pet | noise_ct | noise_both | pet_removal
    level: float         # severity (factor, sigma, or 0/1); 0 or 1 meaning "clean" where applicable
    target: str = ""     # for noise: pet | ct | both

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "family": self.family,
                "level": self.level, "target": self.target}


def specs_from_config(cfg: dict) -> list[Spec]:
    """Expand the perturbation section of the config into the full, ordered list of conditions.

    The clean condition is always first. Levels that equal "no change" (factor 1, sigma 0)
    are NOT repeated per family: they are the clean condition.
    """
    p = cfg["perturbations"]
    specs = [Spec("clean", "clean", "clean", 0.0)]
    for f in p["lowres_factors"]:
        if f != 1:
            specs.append(Spec(f"lowres_x{f:g}", "lowres", "lowres", float(f)))
    for t in p["noise_targets"]:
        for s in p["noise_sigmas"]:
            if s != 0:
                specs.append(Spec(f"noise_{t}_s{s:g}", "noise", f"noise_{t}", float(s), target=t))
    if "removed" in p["pet_removal"]:
        specs.append(Spec("pet_removed", "pet_removed", "pet_removal", 1.0))
    return specs


def apply_spec(spec: Spec, image, case_id: str) -> torch.Tensor:
    """Apply one condition to a clean 2-channel image. Always returns a new tensor."""
    if spec.kind == "clean":
        return _as_float_tensor(image)
    if spec.kind == "lowres":
        return low_resolution(image, spec.level)
    if spec.kind == "noise":
        return gaussian_noise(image, spec.level, spec.target, case_id, name=spec.name)
    if spec.kind == "pet_removed":
        return remove_pet(image)
    raise ValueError(f"unknown perturbation kind: {spec.kind}")


def apply_to_sample(spec: Spec, sample: dict, case_id: str) -> dict:
    """Perturb sample["image"]; the mask entry is passed through untouched (same object)."""
    out = dict(sample)
    out["image"] = apply_spec(spec, sample["image"], case_id)
    return out


# ----------------------------------------------------------------------------- freezing
def fingerprint(specs: Iterable[Spec]) -> str:
    """SHA-256 over the ordered, canonical JSON of all conditions."""
    payload = json.dumps([s.to_dict() for s in specs], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def frozen_record(specs: Sequence[Spec]) -> dict:
    return {"fingerprint": fingerprint(specs), "conditions": [s.to_dict() for s in specs]}

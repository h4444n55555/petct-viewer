"""autoPET PET/CT viewer: a fast, self-contained local web app.

    python viewer.py                 pick a folder in a dialog (or use the buttons in the page)
    python viewer.py D:\\autopet      or pass a folder / files on the command line

What it understands (found by scanning, nothing to configure):
  * nnU-Net layout   imagesTr/<tracer>_<patient>_<study>_0000.nii.gz (CT), ..._0001.nii.gz (PET),
                     labelsTr/<same name>.nii.gz (lesion mask)
  * autoPET original <root>/PETCT_<patient>/<study>/SUV.nii.gz, CTres.nii.gz, SEG.nii.gz
  * any other folder holding one PET file (name contains "suv" or "pet"), one CT file ("ct") and
    optionally one mask ("seg", "mask" or "label")

Why it is fast: a study is decoded from .nii.gz ONCE, reoriented to RAS and cached as plain arrays
(PET float16 SUV, CT int16 HU, mask uint8) under ./cache. After that, opening a study is one np.load and
every slice is a numpy index plus a small PNG. Perturbations come from perturb.py (same code and seeds as
the evaluation) and are computed once per study/condition and cached the same way.

Needs: numpy, scipy, nibabel, pillow, matplotlib (+ torch only for the perturbation conditions).
Everything runs on 127.0.0.1; no data leaves the machine.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
UI_FILE = HERE / "ui.html"

FROZEN = json.loads((HERE / "perturbations_frozen.json").read_text(encoding="utf-8"))
CONDS = FROZEN["conditions"]
COND_BY_NAME = {c["name"]: c for c in CONDS}

# Model-input ranges of the PETCT-RobustSeg study (PET SUV 0..20, CT HU -100..250 are scaled to 0..1).
PET_LO, PET_HI = 0.0, 20.0
CT_LO, CT_HI = -100.0, 250.0

CACHE = Path(os.environ.get("AUTOPET_VIEWER_CACHE") or HERE / "cache")
CACHE_CAP_GB = float(os.environ.get("AUTOPET_VIEWER_CACHE_GB", "20"))
MASK_RGB = np.array([0, 230, 255], np.uint8)          # cyan, readable on both gray and hot colormaps

NII_EXT = (".nii.gz", ".nii")
TORCH_HINT = ("Perturbations need PyTorch, which is not installed here. "
              "Run: pip install torch   (the CPU build is enough), then restart the viewer.")


# =========================================================================== finding studies
def _is_nii(name: str) -> bool:
    return name.lower().endswith(NII_EXT)


def _strip(name: str) -> str:
    for e in NII_EXT:
        if name.lower().endswith(e):
            return name[: -len(e)]
    return name


def _collect(paths) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            for root, dirs, fs in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("cache", "__pycache__")]
                files += [Path(root) / f for f in fs if _is_nii(f)]
        elif p.is_file() and _is_nii(p.name):
            files.append(p)
    return files


_NN = re.compile(r"^(?P<base>.+)_000(?P<ch>\d)$")
_MASK_WORDS = {"seg", "segmentation", "mask", "label", "labels", "lesion", "lesions", "gt"}
_PET_WORDS, _CT_WORDS = {"suv", "pet", "pt"}, {"ct", "ctres"}


def _find_mask(d: Path, base: str, known: set[str]) -> Path | None:
    """The mask of an nnU-Net study: <base>.nii.gz in the sibling labels folder (or next to the images)."""
    cands = [d.parent / "labelsTr", d.parent / d.name.replace("images", "labels"), d]
    for cd in cands:
        for e in NII_EXT:
            f = cd / (base + e)
            if str(f) in known:                                   # found by the scan: no extra disk access
                return f
    if len(known) < 500:                                          # few files chosen by hand: the labels folder was not scanned
        for cd in cands:
            for e in NII_EXT:
                f = cd / (base + e)
                if f.is_file():
                    return f
    return None


def _group(files: list[Path]) -> list[dict]:
    """Turn a flat list of NIfTI files into studies (PET + CT + optional mask)."""
    by_dir: dict[Path, list[Path]] = {}
    for f in files:
        by_dir.setdefault(f.parent, []).append(f)
    known = {str(f) for f in files}
    out: list[dict] = []
    for d, fs in sorted(by_dir.items()):
        low = {f.name.lower(): f for f in fs}
        used: set[Path] = set()
        # 1) autoPET original: SUV / CTres / SEG in one study folder
        pet = low.get("suv.nii.gz") or low.get("suv.nii")
        ct = low.get("ctres.nii.gz") or low.get("ctres.nii")
        if pet and ct:
            seg = low.get("seg.nii.gz") or low.get("seg.nii")
            pid = d.parent.name[6:] if d.parent.name.upper().startswith("PETCT_") else d.parent.name
            out.append(dict(pet=pet, ct=ct, mask=seg, patient_id=pid, study=d.name, tracer="",
                            case_id=f"{pid}/{d.name}"))
            used |= {pet, ct} | ({seg} if seg else set())
        # 2) nnU-Net: <base>_0000 (CT) and <base>_0001 (PET)
        groups: dict[str, dict[str, Path]] = {}
        for f in fs:
            m = _NN.match(_strip(f.name))
            if f not in used and m and m["ch"] in "01":
                groups.setdefault(m["base"], {})[m["ch"]] = f
        for base, g in sorted(groups.items()):
            if "0" not in g or "1" not in g:
                out.append(dict(incomplete=True, pet=g.get("1"), ct=g.get("0"), mask=None, case_id=base))
                continue
            tracer, _, rest = base.partition("_")
            if tracer.lower() in ("fdg", "psma") and "_" in rest:
                pid, study = rest.split("_", 1)
            else:
                tracer, pid, study = "", base, ""
            out.append(dict(pet=g["1"], ct=g["0"], mask=_find_mask(d, base, known), patient_id=pid, study=study,
                            tracer=tracer.lower(), case_id=f"{pid}/{study}" if study else pid))
            used |= {g["0"], g["1"]}
        # 3) anything else: one PET-like and one CT-like file in a folder
        if d.name.lower().startswith("labels"):             # mask-only folders (nnU-Net labelsTr) are not studies
            continue
        rest_files = [f for f in fs if f not in used and not _NN.match(_strip(f.name))]
        cls = {"pet": [], "ct": [], "mask": []}
        for f in rest_files:
            toks = set(re.split(r"[^a-z0-9]+", _strip(f.name).lower()))
            kind = ("mask" if toks & _MASK_WORDS else "pet" if toks & _PET_WORDS else
                    "ct" if toks & _CT_WORDS else None)
            if kind:
                cls[kind].append(f)
        if len(cls["pet"]) == 1 and len(cls["ct"]) == 1:
            out.append(dict(pet=cls["pet"][0], ct=cls["ct"][0], mask=(cls["mask"] or [None])[0],
                            patient_id=d.name, study="", tracer="", case_id=d.name))
        elif (len(cls["pet"]) == 1) != (len(cls["ct"]) == 1) and not (len(cls["pet"]) > 1 or len(cls["ct"]) > 1):
            out.append(dict(incomplete=True, pet=(cls["pet"] or [None])[0], ct=(cls["ct"] or [None])[0],
                            mask=None, case_id=d.name))
    return out


def _read_metadata(dirs) -> dict[str, str]:
    """patient id -> diagnosis, from an autoPET *_metadata.csv sitting next to the data."""
    diag: dict[str, str] = {}
    for d in dirs:
        for f in sorted(d.glob("*metadata*.csv")):
            try:
                with open(f, newline="", encoding="utf-8-sig") as fh:
                    for row in csv.DictReader(fh):
                        sid, dx = row.get("Subject ID", ""), (row.get("diagnosis") or "").strip()
                        if sid and dx:
                            diag.setdefault(sid.split("_", 1)[-1], dx)
            except Exception:                                     # noqa: BLE001  (metadata is optional)
                pass
    return diag


def _read_splits(dirs) -> dict[str, str]:
    """patient id -> train/val/test, from an optional splits.json ({"patients": {"train": [...], ...}})."""
    for d in list(dirs) + [HERE]:
        f = d / "splits.json"
        if f.is_file():
            try:
                sp = json.loads(f.read_text(encoding="utf-8"))["patients"]
                return {p: k for k, v in sp.items() for p in v}
            except Exception:                                     # noqa: BLE001
                pass
    return {}


# =========================================================================== state
CASES: list[dict] = []
SRC = {"label": "", "skipped": 0, "gen": 0}
_state_lock = threading.Lock()
_index: dict[str, dict] = {}                                      # persisted per-study stats, key -> dict
_index_lock = threading.Lock()
IDX = {"running": False, "done": 0, "total": 0}
NO_INDEX = False


def _index_file() -> Path:
    return CACHE / "index.json"


def _load_index() -> None:
    try:
        _index.update(json.loads(_index_file().read_text(encoding="utf-8")))
    except Exception:                                             # noqa: BLE001
        pass


def _save_index() -> None:
    with _index_lock:
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = _index_file().with_suffix(".tmp")
            tmp.write_text(json.dumps(_index), encoding="utf-8")
            os.replace(tmp, _index_file())
        except OSError:
            pass


def _set_stats(c: dict, nx, ny, nz, sp, voxels) -> None:
    c.update(nx=int(nx), ny=int(ny), nz=int(nz), sp=[float(s) for s in sp], lesion_voxels=int(voxels),
             lesion_ml=round(int(voxels) * float(np.prod(sp)) / 1000.0, 2))
    with _index_lock:
        _index[c["key"]] = {"nx": c["nx"], "ny": c["ny"], "nz": c["nz"], "sp": c["sp"], "lesion_voxels": c["lesion_voxels"]}


def load_sources(paths, picked: bool = False) -> dict:
    """Scan folders / files and make them the current study list. Raises ValueError if nothing usable."""
    paths = [str(Path(p).expanduser()) for p in paths]
    files = _collect(paths)
    if picked:                                                    # files chosen one by one: look at their folders too
        chosen = {f.resolve() for f in files}
        for d in {f.parent for f in files}:
            files += [x for x in d.iterdir() if x.is_file() and _is_nii(x.name)]
        files = list({f.resolve(): f for f in files}.values())
    groups = _group(files)
    if picked:                                                    # only studies that contain a chosen file
        groups = [g for g in groups if chosen & {Path(g[k]).resolve() for k in ("pet", "ct", "mask") if g.get(k)}]
    ok = [g for g in groups if not g.get("incomplete")]
    skipped = len(groups) - len(ok)
    if not ok:
        raise ValueError(
            "No PET/CT studies found. Expected either the nnU-Net layout (imagesTr/*_0000.nii.gz = CT, "
            "*_0001.nii.gz = PET, labelsTr/*.nii.gz = mask), the autoPET layout (PETCT_<id>/<study>/SUV.nii.gz, "
            "CTres.nii.gz, SEG.nii.gz), or a folder with one PET and one CT file."
            + (f" ({skipped} incomplete study/studies had only a PET or only a CT file.)" if skipped else ""))

    dirs = []
    for g in ok:
        for d in (g["pet"].parent, g["pet"].parent.parent, g["pet"].parent.parent.parent):
            if d not in dirs:
                dirs.append(d)
    diag, split = _read_metadata(dirs), _read_splits(dirs)

    ok.sort(key=lambda g: (g["patient_id"], g["study"], g["tracer"]))
    counts: dict[str, int] = {}
    for g in ok:
        counts[g["patient_id"]] = counts.get(g["patient_id"], 0) + 1
    cases = []
    for i, g in enumerate(ok):
        key = hashlib.sha1(os.path.abspath(g["pet"]).encode()).hexdigest()[:12]   # no disk access: fast on slow drives
        c = dict(i=i, key=key, pet=str(g["pet"]), ct=str(g["ct"]), mask=str(g["mask"]) if g["mask"] else None,
                 patient_id=g["patient_id"], study=g["study"], tracer=g["tracer"], case_id=g["case_id"],
                 diagnosis=diag.get(g["patient_id"], "n/a"), split=split.get(g["patient_id"], "none"),
                 studies_for_patient=counts[g["patient_id"]],
                 nx=None, ny=None, nz=None, sp=None, lesion_voxels=None, lesion_ml=None)
        e = _index.get(key)
        if e and e.get("nz") is not None:
            _set_stats(c, e["nx"], e["ny"], e["nz"], e["sp"], e["lesion_voxels"])
        cases.append(c)

    roots = [Path(p) for p in paths]
    label = f"{len(roots)} selected file(s) in {roots[0].parent}" if picked else str(roots[0])
    with _state_lock:
        _reset_runtime()
        CASES[:] = cases
        SRC.update(label=label, skipped=skipped, gen=int(time.time() * 1000))
        gen = SRC["gen"]
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "last_source.json").write_text(json.dumps({"paths": paths, "picked": picked}), encoding="utf-8")
    except OSError:
        pass
    if not NO_INDEX:
        threading.Thread(target=_indexer, args=(gen, cases), daemon=True).start()
    return {"count": len(cases), "skipped": skipped, "label": label}


def _last_dir() -> str:
    try:
        p = json.loads((CACHE / "last_source.json").read_text(encoding="utf-8"))["paths"][0]
        return str(Path(p) if Path(p).is_dir() else Path(p).parent)
    except Exception:                                             # noqa: BLE001
        return ""


# =========================================================================== background statistics
def ras_geometry(img):
    """(orientation transform, RAS-ordered shape, RAS-ordered voxel spacing) of a nibabel image."""
    from nibabel.orientations import io_orientation

    ornt = io_orientation(img.affine)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    shape, spacing = [0] * 3, [0.0] * 3
    for ax in range(3):
        k = int(ornt[ax, 0])
        shape[k], spacing[k] = int(img.shape[ax]), zooms[ax]
    return ornt, shape, spacing


def _indexer(gen: int, cases: list[dict]) -> None:
    """Header + lesion-voxel statistics for every study, one at a time, so lists and charts fill in."""
    import nibabel as nib

    todo = [c for c in cases if c["nz"] is None]
    IDX.update(running=True, done=0, total=len(todo))
    try:
        for n, c in enumerate(todo, 1):
            if gen != SRC["gen"]:
                return
            try:
                img = nib.load(c["pet"])
                _, shape, sp = ras_geometry(img)
                vox = 0
                if c["mask"]:
                    m = nib.load(c["mask"])
                    if tuple(m.shape) == tuple(img.shape):
                        vox = int(np.count_nonzero(np.asanyarray(m.dataobj) > 0.5))
                _set_stats(c, *shape, sp, vox)
            except Exception:                                     # noqa: BLE001  (a bad file must not stop the rest)
                traceback.print_exc()
            IDX["done"] = n
            if n % 25 == 0:
                _save_index()
    finally:
        if gen == SRC["gen"]:
            IDX["running"] = False
            _save_index()


# =========================================================================== disk cache
def case_dir(i: int) -> Path:
    return CACHE / CASES[i]["key"]


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def evict(protect: Path | None = None) -> None:
    """Keep the cache under the cap by deleting the least recently used study folders."""
    if not CACHE.exists():
        return
    dirs = [d for d in CACHE.iterdir() if d.is_dir()]
    sizes = {d: dir_size(d) for d in dirs}
    total = sum(sizes.values())
    cap = CACHE_CAP_GB * 1e9
    for d in sorted(dirs, key=lambda d: (d / ".used").stat().st_mtime if (d / ".used").exists() else 0):
        if total <= cap:
            break
        if protect is not None and d == protect:
            continue
        shutil.rmtree(d, ignore_errors=True)
        total -= sizes[d]


def touch(d: Path) -> None:
    (d / ".used").write_text(str(time.time()))


def save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")          # np.save would append .npy to a different name
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


# =========================================================================== case loading
class Case:
    def __init__(self, i, pet, ct, mask, meta):
        self.i, self.pet, self.ct, self.mask, self.meta = i, pet, ct, mask, meta
        self.shape = pet.shape
        self.spacing = meta["spacing"]
        self.mip_cache: dict = {}
        self.cond_cache: dict = {}
        self.lock = threading.Lock()


def _convert(i: int, d: Path, src: list) -> None:
    """Decode the NIfTI files once and write the viewer cache (RAS orientation)."""
    import nibabel as nib
    from nibabel.orientations import apply_orientation

    c = CASES[i]
    pet_img, ct_img = nib.load(c["pet"]), nib.load(c["ct"])
    m_img = nib.load(c["mask"]) if c["mask"] else None
    ornt, _, spacing = ras_geometry(pet_img)                  # the PET affine defines the grid for all three
    warnings = []
    if ct_img.shape != pet_img.shape or not np.allclose(ct_img.affine, pet_img.affine, atol=1e-2):
        warnings.append("CT grid differs from PET grid")

    def ras(img, dtype, fn):
        a = apply_orientation(np.asanyarray(img.dataobj), ornt)
        return np.ascontiguousarray(fn(a).astype(dtype))

    pet = ras(pet_img, np.float16, lambda a: a)
    ct = ras(ct_img, np.int16, lambda a: np.clip(np.rint(a), -32768, 32767))
    if m_img is None:
        mask = np.zeros(pet.shape, np.uint8)
        warnings.append("no mask file for this study")
    elif m_img.shape == pet_img.shape:
        mask = ras(m_img, np.uint8, lambda a: a > 0.5)
        if not np.allclose(m_img.affine, pet_img.affine, atol=1e-2):
            warnings.append("mask header differs from the image header (image grid adopted"
                            + ("; mask is empty so this is harmless)" if not mask.any()
                               else "; mask is NOT empty - check orientation!)"))
    else:
        mask = np.zeros(pet.shape, np.uint8)
        warnings.append(f"mask shape {m_img.shape} differs from image shape {pet_img.shape}; mask not shown")
    d.mkdir(parents=True, exist_ok=True)
    save_npy(d / "pet.npy", pet)
    save_npy(d / "ct.npy", ct)
    save_npy(d / "mask.npy", mask)
    area = mask.sum(axis=(0, 1))
    z0 = int(area.argmax()) if area.max() > 0 else pet.shape[2] // 2
    if area.max() > 0:
        xs, ys = np.nonzero(mask[:, :, z0])
        x0, y0 = int(xs.mean()), int(ys.mean())
    else:
        x0, y0 = pet.shape[0] // 2, pet.shape[1] // 2
    meta = {"shape": list(pet.shape), "spacing": spacing, "focus": [x0, y0, z0], "warnings": warnings,
            "orientation": "".join(nib.aff2axcodes(pet_img.affine)),
            "pet_max": float(pet.max()), "ct_min": int(ct.min()), "ct_max": int(ct.max()),
            "mask_voxels": int(mask.sum()), "src": src}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


_cases: dict[int, Case] = {}
_cases_lock = threading.Lock()
_load_locks: dict[int, threading.Lock] = {}
_compute_lock = threading.Lock()
_cond_locks: dict = {}
_job = {"running": False, "case": None, "current": None, "done": 0, "total": 0, "error": None}


def _reset_runtime() -> None:
    """Forget everything tied to the previous study list (called when a new folder is loaded)."""
    with _cases_lock:
        _cases.clear()
        _load_locks.clear()
    _cond_locks.clear()
    _job.update(running=False, case=None, current=None, done=0, total=0, error=None)


def get_case(i: int) -> Case:
    if not 0 <= i < len(CASES):
        raise IndexError("unknown study (the list changed; reload the page)")
    with _cases_lock:
        if i in _cases:
            c = _cases.pop(i)
            _cases[i] = c                                   # most recently used last
            return c
        lock = _load_locks.setdefault(i, threading.Lock())
    with lock:
        with _cases_lock:
            if i in _cases:
                return _cases[i]
        d = case_dir(i)
        st = os.stat(CASES[i]["pet"])
        src = [st.st_size, st.st_mtime_ns]
        if (d / "meta.json").exists():                       # a cached copy of a file that has since changed is stale
            try:
                stale = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("src") != src
            except ValueError:
                stale = True
            if stale:
                shutil.rmtree(d, ignore_errors=True)
        if not (d / "meta.json").exists():
            _convert(i, d, src)
        touch(d)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        c = Case(i, np.load(d / "pet.npy"), np.load(d / "ct.npy"), np.load(d / "mask.npy"), meta)
        X, Y, Z = meta["shape"]
        _set_stats(CASES[i], X, Y, Z, meta["spacing"], meta["mask_voxels"])
        with _cases_lock:
            _cases[i] = c
            while len(_cases) > 2:
                _cases.pop(next(iter(_cases)))
        evict(protect=d)
        return c


# =========================================================================== perturbations
def cond_path(i: int, name: str) -> Path:
    return case_dir(i) / f"cond_{name}.npy"


def cached_conds(i: int) -> list[str]:
    d = case_dir(i)
    return [c["name"] for c in CONDS if c["name"] == "clean" or (d / f"cond_{c['name']}.npy").exists()]


def get_cond(i: int, name: str):
    """(2, X, Y, Z) float16 in the model-input scale [PET, CT], produced by perturb.py (cached)."""
    if name == "clean":
        return None
    case = get_case(i)
    key = (i, name)
    with case.lock:
        if name in case.cond_cache:
            return case.cond_cache[name]
    p = cond_path(i, name)
    lock = _cond_locks.setdefault(key, threading.Lock())
    with lock:
        if not p.exists():
            if importlib.util.find_spec("torch") is None:
                raise RuntimeError(TORCH_HINT)
            with _compute_lock:                            # one heavy computation at a time
                import torch
                import perturb

                img = np.empty((2,) + case.shape, np.float32)
                img[0] = np.clip(case.pet.astype(np.float32), PET_LO, PET_HI)
                img[0] = (img[0] - PET_LO) / (PET_HI - PET_LO)
                img[1] = (np.clip(case.ct.astype(np.float32), CT_LO, CT_HI) - CT_LO) / (CT_HI - CT_LO)
                spec = perturb.Spec(**COND_BY_NAME[name])
                out = perturb.apply_spec(spec, torch.from_numpy(img), CASES[i]["case_id"]).numpy()
                del img
                save_npy(p, out.astype(np.float16))
                touch(case_dir(i))
        vol = np.load(p, mmap_mode="r")
    with case.lock:
        case.cond_cache[name] = vol
        while len(case.cond_cache) > 4:
            case.cond_cache.pop(next(iter(case.cond_cache)))
    return vol


def _run_all(i: int) -> None:
    try:
        todo = [c["name"] for c in CONDS if c["name"] != "clean"]
        _job.update(running=True, case=i, done=0, total=len(todo), error=None)
        for n in todo:
            _job["current"] = n
            get_cond(i, n)
            _job["done"] += 1
        evict(protect=case_dir(i))
    except Exception as e:                                  # noqa: BLE001
        _job["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _job.update(running=False, current=None)


# =========================================================================== lesions
def lesions_for(i: int) -> dict:
    d = case_dir(i)
    f = d / "lesions.json"
    case = get_case(i)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    from scipy import ndimage

    dx, dy, dz = case.spacing
    vox_ml = dx * dy * dz / 1000.0
    out = {"count": 0, "total_ml": 0.0, "lesions": []}
    if case.mask.any():
        lab, n = ndimage.label(case.mask, structure=np.ones((3, 3, 3), bool))   # 26-connectivity
        sizes = np.bincount(lab.ravel())[1:]
        boxes = ndimage.find_objects(lab)
        items = []
        for k, sl in enumerate(boxes, start=1):
            sub = lab[sl] == k
            suv = case.pet[sl].astype(np.float32)[sub]
            loc = np.argwhere(sub)
            peak = loc[int(suv.argmax())] + np.array([s.start for s in sl])
            ext = [(s.stop - s.start) * sp for s, sp in zip(sl, case.spacing)]
            items.append({"voxels": int(sizes[k - 1]), "ml": round(float(sizes[k - 1] * vox_ml), 3),
                          "suv_max": round(float(suv.max()), 2), "suv_mean": round(float(suv.mean()), 2),
                          "peak": [int(v) for v in peak], "extent_mm": [round(float(e), 1) for e in ext]})
        items.sort(key=lambda r: -r["voxels"])
        for j, r in enumerate(items, start=1):
            r["id"] = j
        out = {"count": n, "total_ml": round(float(case.mask.sum() * vox_ml), 2), "lesions": items[:150]}
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


# =========================================================================== rendering
def _luts() -> dict:
    from matplotlib import colormaps

    t = np.linspace(0, 1, 256)
    return {n: (colormaps[m](t)[:, :3] * 255).astype(np.uint8)
            for n, m in (("magma", "magma"), ("hot", "hot"), ("gray_r", "gray_r"), ("viridis", "viridis"),
                         ("diff", "RdBu_r"))}


LUTS = _luts()


def disp(a: np.ndarray) -> np.ndarray:
    """RAS array slice (first axis, second axis) -> display image: rows run superior/anterior at the top,
    columns run patient-right on the left (radiological convention)."""
    return a[::-1, ::-1].T


def take(vol, plane: str, idx: int):
    if plane == "ax":
        return vol[:, :, idx]
    if plane == "co":
        return vol[:, idx, :]
    return vol[idx, :, :]


def mip_of(case: Case, cond: str, plane: str):
    """Maximum-intensity projection of PET (display units) and of the mask, cached per condition."""
    key = (cond, plane)
    axis = 1 if plane == "mipco" else 0
    with case.lock:
        if key in case.mip_cache:
            return case.mip_cache[key]
    if cond == "clean":
        pet = case.pet.max(axis=axis).astype(np.float32)
    else:
        vol = get_cond(case.i, cond)
        pet = np.asarray(vol[0]).max(axis=axis).astype(np.float32) * (PET_HI - PET_LO) + PET_LO
    msk = case.mask.max(axis=axis).astype(bool)
    with case.lock:
        case.mip_cache[key] = (pet, msk)
    return pet, msk


def units(case: Case, cond: str, plane: str, idx: int, clip: bool):
    """2D PET (SUV-equivalent) and CT (HU-equivalent) slices in display orientation, plus the mask slice."""
    if plane in ("mipco", "mipsa"):
        pet, msk = mip_of(case, cond, plane)
        if cond == "clean" and clip:
            pet = np.clip(pet, PET_LO, PET_HI)
        return disp(pet), None, disp(msk)
    msk = disp(take(case.mask, plane, idx)).astype(bool)
    if cond == "clean":
        pet = disp(take(case.pet, plane, idx)).astype(np.float32)
        ct = disp(take(case.ct, plane, idx)).astype(np.float32)
        if clip:
            pet, ct = np.clip(pet, PET_LO, PET_HI), np.clip(ct, CT_LO, CT_HI)
    else:
        vol = get_cond(case.i, cond)
        pet = disp(take(vol[0], plane, idx)).astype(np.float32) * (PET_HI - PET_LO) + PET_LO
        ct = disp(take(vol[1], plane, idx)).astype(np.float32) * (CT_HI - CT_LO) + CT_LO
    return pet, ct, msk


def outline(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def render(q: dict) -> bytes:
    i, plane, cond = int(q["i"]), q.get("plane", "ax"), q.get("cond", "clean")
    idx = int(q.get("idx", 0))
    view, cmap = q.get("view", "fused"), q.get("cmap", "magma")
    phi, clo, chi = float(q.get("phi", 8)), float(q.get("clo", CT_LO)), float(q.get("chi", CT_HI))
    alpha, maskmode, clip = float(q.get("alpha", 0.6)), q.get("mask", "outline"), q.get("clip", "0") == "1"
    case = get_case(i)
    if view == "diff":
        ch, dlim = q.get("ch", "pet"), max(float(q.get("dlim", 0.1)), 1e-4)
        pet, ct, msk = units(case, cond, plane, idx, True)
        pet0, ct0, _ = units(case, "clean", plane, idx, True)
        d = (pet - pet0) / (PET_HI - PET_LO) if ch == "pet" else (ct - ct0) / (CT_HI - CT_LO)
        rgb = LUTS["diff"][np.clip(((d / dlim) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)]
    else:
        pet, ct, msk = units(case, cond, plane, idx, clip)
        pet_n = np.clip((pet - PET_LO) / max(phi - PET_LO, 1e-6), 0, 1)
        pet_rgb = LUTS[cmap][(pet_n * 255).astype(np.uint8)]
        if ct is None or view == "pet":
            rgb = pet_rgb
        else:
            g = (np.clip((ct - clo) / max(chi - clo, 1e-6), 0, 1) * 255).astype(np.uint8)
            ct_rgb = np.repeat(g[..., None], 3, axis=2)
            if view == "ct":
                rgb = ct_rgb
            else:                                             # fused: PET opacity grows with uptake
                a = (alpha * np.clip(pet_n / 0.15, 0, 1))[..., None]
                rgb = (ct_rgb * (1 - a) + pet_rgb * a).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    if maskmode != "off" and msk is not None and msk.any():
        if maskmode == "fill":
            rgb[msk] = (rgb[msk] * 0.6 + MASK_RGB * 0.4).astype(np.uint8)
        rgb[outline(msk)] = MASK_RGB
    if q.get("flip", "0") == "1":
        rgb = rgb[:, ::-1]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def probe(q: dict) -> dict:
    i, x, y, z = int(q["i"]), int(q["x"]), int(q["y"]), int(q["z"])
    case = get_case(i)
    X, Y, Z = case.shape
    x, y, z = min(max(x, 0), X - 1), min(max(y, 0), Y - 1), min(max(z, 0), Z - 1)
    out = {"suv": round(float(case.pet[x, y, z]), 2), "hu": int(case.ct[x, y, z]), "lesion": bool(case.mask[x, y, z])}
    cond = q.get("cond", "clean")
    if cond != "clean":
        vol = get_cond(i, cond)
        out["suv_cond"] = round(float(vol[0][x, y, z]) * (PET_HI - PET_LO) + PET_LO, 2)
        out["hu_cond"] = int(round(float(vol[1][x, y, z]) * (CT_HI - CT_LO) + CT_LO))
    return out


# =========================================================================== file / folder dialog
_PICK_CODE = r"""
import json, sys
import tkinter as tk
from tkinter import filedialog
mode, start = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
root.update()
kw = {"initialdir": start} if start else {}
if mode == "folder":
    p = filedialog.askdirectory(parent=root, title="Choose the folder with your autoPET studies", **kw)
    out = [p] if p else []
else:
    out = list(filedialog.askopenfilenames(parent=root, title="Choose NIfTI files (PET, CT and, optionally, mask)",
                                           filetypes=[("NIfTI", "*.nii.gz *.nii"), ("All files", "*.*")], **kw))
print(json.dumps(out))
"""


def pick(mode: str) -> list[str]:
    """Open a native folder / file dialog (in a subprocess so it can never hang the server)."""
    try:
        r = subprocess.run([sys.executable, "-c", _PICK_CODE, mode, _last_dir()],
                           capture_output=True, text=True, timeout=900)
    except Exception as e:                                       # noqa: BLE001
        raise RuntimeError(f"could not open a dialog ({type(e).__name__})") from e
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise RuntimeError(f"no file dialog available on this system ({last})")
    return json.loads(r.stdout.strip().splitlines()[-1])


# =========================================================================== http
def cases_payload() -> list[dict]:
    try:
        have = {d.name for d in CACHE.iterdir() if d.is_dir()}
    except OSError:
        have = set()
    keys = ("i", "patient_id", "study", "tracer", "diagnosis", "split", "lesion_ml", "lesion_voxels", "nz",
            "studies_for_patient", "case_id")
    return [{**{k: c[k] for k in keys}, "cached": c["key"] in have and (CACHE / c["key"] / "meta.json").exists()}
            for c in list(CASES)]


def init_payload() -> dict:
    return {"cases": cases_payload(), "conds": CONDS, "pet_hi": PET_HI, "ct_lo": CT_LO, "ct_hi": CT_HI,
            "root": SRC["label"], "skipped": SRC["skipped"], "gen": SRC["gen"], "cache": str(CACHE),
            "cache_cap_gb": CACHE_CAP_GB, "fingerprint": FROZEN.get("fingerprint", "")[:12],
            "indexing": dict(IDX), "torch": importlib.util.find_spec("torch") is not None}


class Handler(BaseHTTPRequestHandler):
    server_version = "autopet-viewer"

    def log_message(self, *a):                              # quiet
        pass

    def _send(self, code, body: bytes, ctype: str, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):                                       # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif u.path == "/slice":
                self._send(200, render(q), "image/png", cache="private, max-age=3600")
            elif u.path == "/probe":
                self._json(probe(q))
            elif u.path == "/api/init":
                self._json(init_payload())
            elif u.path == "/api/cases":                    # polled while statistics are still being collected
                self._json({"cases": cases_payload(), "indexing": dict(IDX), "gen": SRC["gen"]})
            elif u.path == "/api/pick":                     # native dialog, then scan what was chosen
                mode = q.get("mode", "folder")
                try:
                    chosen = pick(mode)
                except RuntimeError as e:
                    self._json({"error": str(e), "no_dialog": True}, 500)
                    return
                if not chosen:
                    self._json({"cancelled": True})
                    return
                self._json({"ok": True, **load_sources(chosen, picked=(mode == "files"))})
            elif u.path == "/api/load":                     # a path typed or pasted into the page
                p = q.get("path", "").strip().strip('"')
                if not p or not Path(p).expanduser().exists():
                    self._json({"error": f"path not found: {p or '(empty)'}"}, 400)
                    return
                self._json({"ok": True, **load_sources([p])})
            elif u.path == "/api/open":
                i = int(q["i"])
                t0 = time.time()
                c = get_case(i)
                self._json({**c.meta, "seconds": round(time.time() - t0, 2), "cached_conds": cached_conds(i)})
            elif u.path == "/api/lesions":
                self._json(lesions_for(int(q["i"])))
            elif u.path == "/api/cond":
                t0 = time.time()
                get_cond(int(q["i"]), q["name"])
                self._json({"ok": True, "seconds": round(time.time() - t0, 2), "cached_conds": cached_conds(int(q["i"]))})
            elif u.path == "/api/cond_all":
                i = int(q["i"])
                if importlib.util.find_spec("torch") is None:
                    self._json({"error": TORCH_HINT}, 500)
                    return
                if not _job["running"]:
                    threading.Thread(target=_run_all, args=(i,), daemon=True).start()
                    time.sleep(0.05)
                self._json(dict(_job))
            elif u.path == "/api/cond_status":
                self._json({**_job, "cached_conds": cached_conds(int(q["i"]))})
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:                              # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global CACHE, CACHE_CAP_GB, NO_INDEX
    ap = argparse.ArgumentParser(description="autoPET PET/CT viewer. Run it, then choose a folder or files.")
    ap.add_argument("paths", nargs="*", help="folder(s) or NIfTI files to open (optional: a dialog appears if omitted)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    ap.add_argument("--no-dialog", action="store_true", help="do not show the folder dialog at start-up")
    ap.add_argument("--no-index", action="store_true", help="skip the background pass that measures lesion volumes")
    ap.add_argument("--cache", help="where decoded studies are cached (default: ./cache next to viewer.py)")
    ap.add_argument("--cache-gb", type=float, help="cache size limit in GB (default 20)")
    a = ap.parse_args()
    if a.cache:
        CACHE = Path(a.cache)
    if a.cache_gb:
        CACHE_CAP_GB = a.cache_gb
    NO_INDEX = a.no_index
    CACHE.mkdir(parents=True, exist_ok=True)
    _load_index()

    paths = a.paths or ([os.environ["AUTOPET_DATA"]] if os.environ.get("AUTOPET_DATA") else [])
    picked = bool(paths) and all(Path(p).is_file() for p in paths)
    if not paths and not a.no_dialog:
        print("Choose the folder that holds your autoPET studies (a dialog should appear)...")
        try:
            paths = pick("folder")
        except RuntimeError as e:
            print(f"  {e}. You can type a folder path in the page instead.")
    if paths:
        try:
            r = load_sources(paths, picked=picked)
            print(f"Found {r['count']} stud{'y' if r['count'] == 1 else 'ies'} in {r['label']}"
                  + (f" ({r['skipped']} skipped: incomplete)" if r["skipped"] else ""))
        except ValueError as e:
            print(f"  {e}")
    if importlib.util.find_spec("torch") is None:
        print("Note: PyTorch is not installed, so the perturbation conditions are unavailable "
              "(everything else works). Install it with: pip install torch")

    srv, port = None, a.port
    for port in range(a.port, a.port + 15):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"Could not open a local port between {a.port} and {a.port + 14}.")
    url = f"http://127.0.0.1:{port}/"
    print(f"PET/CT viewer on {url}   (cache: {CACHE}, cap {CACHE_CAP_GB:g} GB)   Ctrl+C to stop")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _save_index()


if __name__ == "__main__":
    main()

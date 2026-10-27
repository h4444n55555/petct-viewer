# PET/CT Viewer

A local web viewer for PET/CT studies stored as NIfTI. Built for the [autoPET](https://autopet.grand-challenge.org/)
dataset, and works with any PET + CT (+ optional lesion mask) data in the same layout.

Axial / coronal / sagittal views, PET MIP, PET/CT fusion, lesion-mask overlay, hover SUV/HU, lesion list,
robustness perturbations (low resolution, noise, PET removed) and dataset overview charts.
Everything runs on your computer; no data is uploaded.

## Setup

1. Install [Python 3.9+](https://www.python.org/downloads/)
2. In this folder run:

```
pip install -r requirements.txt
python viewer.py
```

3. Choose the folder with your studies in the dialog. The viewer opens in your browser.

Windows shortcut: double-click `start.bat` (does steps 2 and 3 for you). macOS/Linux: `bash start.sh`.

To open something else later, use **Open folder…** or **Choose files…** at the top of the page.
You can also skip the dialog: `python viewer.py D:\path\to\data`.

## Data it understands

Pick any folder; it is scanned for studies automatically.

| Layout | Files |
|---|---|
| nnU-Net / autoPET download | `imagesTr/<name>_0000.nii.gz` (CT), `<name>_0001.nii.gz` (PET), `labelsTr/<name>.nii.gz` (mask) |
| autoPET original | `PETCT_<id>/<study>/SUV.nii.gz`, `CTres.nii.gz`, `SEG.nii.gz` |
| Any other | one folder with a PET file (name contains `suv` or `pet`), a CT file (`ct`) and optionally a mask (`seg`, `mask` or `label`) |

## Good to know

- The first open of a study takes 5-15 s; after that it is instant (decoded copies are kept in `cache/`, 20 GB max, safe to delete).
- Lesion volumes in the list and overview fill in over the first few minutes of a large folder.
- `torch` is only needed for the perturbation views; everything else works without it.
- Keys: `wheel` scroll slices, `[` `]` previous/next study, `n` `p` next/previous lesion, `m` mask, `c` next condition, `x` compare.

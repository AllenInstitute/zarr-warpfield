"""Check the transform chain against the pipeline's own CCF-on-brain labels.

    validate.py SRC ATLAS CACHE [--stride 3]

The AIND pipeline resampled a CCF annotation that had been moved into SPIM
template space onto the brain's reoriented level-3 grid
(registration_metadata/moved_ccf_anno_to_ls.nii.gz, genericLabel
interpolation); the static `ccf_annotation_precomputed` is that volume. On a
subsample of that grid this compares it with:

1. steps 1-2 of `chain.py` (brain -> SPIM template) sampling the CCF
   annotation in SPIM template space from allen-atlas-assets
   (allen-adult-mouse-spim-lca-annotation 2024-05): tests the brain frame
   conventions and the use of the pipeline's ANTs files;
2. the full chain (+ SPIM -> STPT 2020) sampling the CCF 2017 annotation in
   STPT space: the composed transform.

Labels are sampled nearest-neighbour. Reported: agreement (fraction of voxels
labelled in either volume with the same label), Dice of the brain masks, and
Dice per large structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain import (  # noqa: E402
    LEVEL3_SHAPE_XYZ,
    Chain,
    in_chunks,
    lps_affine,
    lps_to_ras_zyx,
    nifti_header_affine,
)

ALIGN = "image_atlas_alignment/Ex_639_Em_667"
BUCKET = (
    "https://aind-open-data.s3.us-west-2.amazonaws.com/"
    "894847_2026-11-20_11-27-00_stitched_2026-11-21_13-35-21/"
)
# STPT 2020 RAS = 2015 RAS + this (template_25 headers of the two years)
RAS_2015_TO_2020 = np.array([-5.695, 5.35, 5.22])


def nearest(img, data, p_lps):
    idx = np.rint(p_lps @ np.linalg.inv(lps_affine(img))[:3, :3].T + np.linalg.inv(lps_affine(img))[:3, 3])
    idx = idx.astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(data.shape[:3])), axis=1)
    out = np.zeros(len(p_lps), dtype=np.int64)
    out[ok] = data[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out


def compare(ref, got, name):
    either = (ref > 0) | (got > 0)
    agree = float(np.mean(ref[either] == got[either]))
    a, b = ref > 0, got > 0
    dice_mask = float(2 * np.sum(a & b) / (a.sum() + b.sum()))
    ids, counts = np.unique(ref[ref > 0], return_counts=True)
    big = ids[np.argsort(counts)[::-1][:12]]
    per = {
        int(i): round(float(2 * np.sum((ref == i) & (got == i)) / (np.sum(ref == i) + np.sum(got == i))), 3)
        for i in big
    }
    print(f"{name}: agreement {agree:.4f}  brain-mask Dice {dice_mask:.4f}  "
          f"median Dice of 12 largest structures {np.median(list(per.values())):.3f}", flush=True)
    return {"agreement": agree, "mask_dice": dice_mask, "largest_structures_dice": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("atlas", type=Path)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--fields", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    al = a.src / ALIGN
    shape, level3 = nifti_header_affine(BUCKET + ALIGN + "/metadata/registration_metadata/prep_zarr_img.nii.gz")
    assert shape[:3] == tuple(LEVEL3_SHAPE_XYZ), shape
    ch = Chain(
        level3,
        al / "ls_to_template_SyN_0GenericAffine.mat",
        None,
        al / "ls_to_template_SyN_1InverseWarp.nii.gz",
        None,
        a.fields / "inverse-transform-SPIM.nii.gz",
        a.cache,
    )
    ref_img = nib.load(al / "metadata/registration_metadata/moved_ccf_anno_to_ls.nii.gz")
    ref = np.asanyarray(ref_img.dataobj)
    s = a.stride
    sub = ref[::s, ::s, ::s]
    abc = np.stack(np.meshgrid(*[np.arange(0, n, s) for n in ref.shape[:3]], indexing="ij"), -1).reshape(-1, 3)
    refv = np.rint(sub.reshape(-1)).astype(np.int64)
    p = abc @ lps_affine(ref_img)[:3, :3].T + lps_affine(ref_img)[:3, 3]
    print(f"{len(p):,} voxels (stride {s}), {np.mean(refv > 0):.1%} labelled", flush=True)

    t = in_chunks(ch.brain_lps_to_template, p)
    spim = nib.load(a.atlas / "allen-adult-mouse-spim-lca-annotation_annotations_compressed_25.nii.gz")
    spim_data = np.asanyarray(spim.dataobj)
    got1 = nearest(spim, np.rint(spim_data).astype(np.int64), t)
    res = {"steps_1_2_vs_pipeline": compare(refv, got1, "steps 1-2 (brain -> SPIM template), atlas-assets CCF in SPIM space")}

    stpt = lps_to_ras_zyx(in_chunks(ch.template_to_stpt, t))  # (S, A, R) 2020
    ras2015 = stpt[:, ::-1] - RAS_2015_TO_2020
    ccf = nib.load(a.atlas / "allen-adult-mouse-annotation_annotations_compressed_25.nii.gz")
    ccf_data = np.asanyarray(ccf.dataobj)
    lps2015 = ras2015 * np.array([-1.0, -1.0, 1.0])
    got2 = nearest(ccf, ccf_data, lps2015)
    res["full_chain_vs_pipeline"] = compare(refv, got2, "full chain (+ your SPIM -> STPT), CCF 2017 annotation")
    res["full_chain_vs_steps_1_2"] = compare(got1, got2, "full chain vs steps 1-2")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

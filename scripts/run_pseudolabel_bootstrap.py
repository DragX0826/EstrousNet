"""
Zero-annotation pseudo-label bootstrap for rat vaginal smear cell segmentation.

Pipeline:
  1) Generate pseudo-masks from classical CV
  2) Train lightweight patch-based Mini U-Net (CPU friendly)
  3) Self-training refinement round
  4) Export overlays + per-image cell statistics

Usage:
  python scripts/run_pseudolabel_bootstrap.py --img_dir data/raw --out_dir results/bootstrap
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage as ndi
from skimage import measure, segmentation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.stage_classifier import infer_stage, load_stage_rules

# ---------------------------- defaults ---------------------------- #
PATCH_SIZE = 64
PATCHES_PER_IMG = 120
CONF_THRESH = 0.62
ADAPTIVE_TOP_PERCENT = 40.0
AREA_MIN = 500
AREA_MAX = 15000
CIRCULARITY_MIN = 0.30
EPOCHS_STAGE1 = 12
EPOCHS_STAGE2 = 8
LR = 3e-4
BATCH_SIZE = 16
BASE_CHANNELS = 32
DEVICE = "cpu"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------- stage 1 ---------------------------- #
def generate_pseudo_mask(
    img_bgr: np.ndarray,
    area_min: int,
    area_max: int,
    circularity_min: float,
) -> np.ndarray:
    """Pseudo-mask focused on whole-cell regions (not only nuclei)."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    margin = 25

    # A) Cytoplasm mask for pale pink/light purple cornified cells.
    cyto_hi = cv2.inRange(hsv, np.array([158, 6, 195]), np.array([179, 80, 255]))
    cyto_lo = cv2.inRange(hsv, np.array([0, 6, 195]), np.array([8, 80, 255]))
    cyto = cv2.bitwise_or(cyto_hi, cyto_lo)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    cyto = cv2.morphologyEx(cyto, cv2.MORPH_CLOSE, kernel21, iterations=4)
    cyto = cv2.morphologyEx(cyto, cv2.MORPH_OPEN, kernel15, iterations=1)
    cyto = ndi.binary_fill_holes(cyto > 0).astype(np.uint8) * 255

    # B) Nucleus mask for dark purple/magenta nuclei.
    nuc_hsv = cv2.inRange(hsv, np.array([120, 40, 20]), np.array([175, 255, 180]))
    nuc_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 60, 80]))
    nucleus = cv2.bitwise_or(nuc_hsv, nuc_dark)
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_OPEN, kernel5, iterations=1)
    nucleus = cv2.morphologyEx(nucleus, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # C) Watershed split for large cytoplasm blobs.
    cyto_labels = measure.label(cyto > 0)
    split_mask = np.zeros((h, w), dtype=np.uint8)
    for region in measure.regionprops(cyto_labels):
        blob = (cyto_labels == region.label).astype(np.uint8)
        if region.area < area_min:
            continue

        if region.area < area_max * 0.6:
            split_mask[blob > 0] = 255
            continue

        minr, minc, maxr, maxc = region.bbox
        pad = 5
        r1, r2 = max(0, minr - pad), min(h, maxr + pad)
        c1, c2 = max(0, minc - pad), min(w, maxc + pad)

        roi_blob = blob[r1:r2, c1:c2]
        roi_nuc = (nucleus[r1:r2, c1:c2] > 0) & (roi_blob > 0)
        dist = ndi.distance_transform_edt(roi_blob)
        seed_map = np.zeros_like(roi_blob, dtype=np.int32)

        nuc_labels, n_seeds = ndi.label(roi_nuc)
        if n_seeds == 0:
            from skimage.feature import peak_local_max

            peaks = peak_local_max(dist, min_distance=20, labels=roi_blob.astype(bool))
            for i, (pr, pc) in enumerate(peaks, start=1):
                seed_map[pr, pc] = i
            n_seeds = len(peaks)
        else:
            seed_map = nuc_labels

        if n_seeds == 0:
            split_mask[r1:r2, c1:c2][roi_blob > 0] = 255
            continue

        bg_seed = n_seeds + 1
        border = np.zeros_like(roi_blob, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        seed_map[border & (roi_blob == 0)] = bg_seed

        ws = segmentation.watershed(-dist, seed_map, mask=roi_blob.astype(bool))
        for label_id in range(1, n_seeds + 1):
            cell_region = (ws == label_id).astype(np.uint8)
            cell_area = int(cell_region.sum())
            if area_min <= cell_area <= area_max:
                split_mask[r1:r2, c1:c2][cell_region > 0] = 255

    # D) Supplement for small nucleated cells.
    nuc_dilated = cv2.dilate(nucleus, kernel5, iterations=2)
    nuc_cells = ndi.binary_fill_holes(nuc_dilated > 0).astype(np.uint8) * 255

    # E) Merge + clean.
    combined = cv2.bitwise_or(split_mask, nuc_cells)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # F) Final filtering.
    out = np.zeros((h, w), dtype=np.uint8)
    labels = measure.label(combined > 0)
    for region in measure.regionprops(labels):
        area = region.area
        if area < area_min or area > area_max:
            continue
        perim = region.perimeter if region.perimeter > 0 else 1.0
        circ = 4 * np.pi * area / (perim**2)
        if circ < circularity_min:
            continue
        minr, minc, maxr, maxc = region.bbox
        if minr < margin or minc < margin or maxr > h - margin or maxc > w - margin:
            continue
        out[labels == region.label] = 255
    return out


# ---------------------------- stage 2 ---------------------------- #
class PatchDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        patch_size: int = PATCH_SIZE,
        patches_per_img: int = PATCHES_PER_IMG,
        augment: bool = True,
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.patch_size = patch_size
        self.augment = augment
        ps = patch_size

        for img_path, mask_path in pairs:
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            h, w = img.shape[:2]
            if h < ps or w < ps:
                continue

            cell_yx = np.argwhere(mask > 127)
            if len(cell_yx) > 0:
                chosen = cell_yx[
                    np.random.choice(
                        len(cell_yx),
                        min(patches_per_img // 2, len(cell_yx)),
                        replace=False,
                    )
                ]
                for cy, cx in chosen:
                    r1 = max(0, int(cy) - ps // 2)
                    c1 = max(0, int(cx) - ps // 2)
                    r2 = min(h, r1 + ps)
                    c2 = min(w, c1 + ps)
                    r1 = r2 - ps
                    c1 = c2 - ps
                    self.samples.append((img[r1:r2, c1:c2].copy(), mask[r1:r2, c1:c2].copy()))

            for _ in range(patches_per_img // 2):
                r1 = random.randint(0, h - ps)
                c1 = random.randint(0, w - ps)
                self.samples.append((img[r1 : r1 + ps, c1 : c1 + ps].copy(), mask[r1 : r1 + ps, c1 : c1 + ps].copy()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, mask = self.samples[idx]
        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MiniUNet(nn.Module):
    """Tiny U-Net, CPU-friendly."""

    def __init__(self, in_ch: int = 3, base: int = 16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


def train_model(
    model: nn.Module,
    pairs: list[tuple[Path, Path]],
    epochs: int,
    desc: str,
    batch_size: int,
    patch_size: int,
    patches_per_img: int,
    lr: float,
) -> nn.Module:
    dataset = PatchDataset(
        pairs=pairs,
        patch_size=patch_size,
        patches_per_img=patches_per_img,
        augment=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{desc}: dataset is empty. Check image/mask generation.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for imgs, masks in tqdm(loader, desc=f"{desc} {ep}/{epochs}", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            pred = model(imgs)
            loss = bce(pred, masks)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        mean_loss = total_loss / max(1, len(loader))
        print(f"[{desc}] epoch {ep:02d} loss={mean_loss:.4f}")
    return model


def _sliding_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def infer_full_image(
    model: nn.Module,
    img_bgr: np.ndarray,
    patch_size: int,
    stride: int = 48,
    roi_mask: np.ndarray | None = None,
    roi_dilate: int = 11,
) -> np.ndarray:
    """Sliding-window full-image inference -> prob map [0,1]. Optional ROI-only mode."""
    model.eval()
    h, w = img_bgr.shape[:2]
    prob = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)

    row_starts = _sliding_starts(h, patch_size, stride)
    col_starts = _sliding_starts(w, patch_size, stride)

    roi = None
    if roi_mask is not None:
        roi = (roi_mask > 0).astype(np.uint8)
        if roi_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (roi_dilate, roi_dilate))
            roi = cv2.dilate(roi, k, iterations=1)

    processed_windows = 0
    with torch.no_grad():
        for r in row_starts:
            for c in col_starts:
                if roi is not None and np.max(roi[r : r + patch_size, c : c + patch_size]) == 0:
                    continue
                patch = img_bgr[r : r + patch_size, c : c + patch_size]
                t = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                p = model(t.to(DEVICE)).squeeze().cpu().numpy()
                prob[r : r + patch_size, c : c + patch_size] += p
                cnt[r : r + patch_size, c : c + patch_size] += 1
                processed_windows += 1

    # Fallback to full-image sweep when ROI contains no valid windows.
    if processed_windows == 0:
        with torch.no_grad():
            for r in row_starts:
                for c in col_starts:
                    patch = img_bgr[r : r + patch_size, c : c + patch_size]
                    t = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                    p = model(t.to(DEVICE)).squeeze().cpu().numpy()
                    prob[r : r + patch_size, c : c + patch_size] += p
                    cnt[r : r + patch_size, c : c + patch_size] += 1

    cnt = np.where(cnt == 0, 1, cnt)
    return prob / cnt


def mask_from_prob(
    prob: np.ndarray,
    conf_floor: float,
    adaptive_top_percent: float,
    area_min: int,
    area_max: int,
) -> tuple[np.ndarray, float]:
    """
    Threshold prob and filter by morphology.

    If adaptive_top_percent > 0:
      keep top-K% confident pixels with a minimum threshold floor.
    """
    threshold = conf_floor
    if adaptive_top_percent > 0:
        flat = prob.reshape(-1)
        percentile = float(np.clip(100.0 - adaptive_top_percent, 0.0, 100.0))
        adaptive_thr = float(np.percentile(flat, percentile))
        threshold = max(conf_floor, adaptive_thr)

    binary = (prob > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    labels = measure.label(binary > 0)
    out = np.zeros_like(binary)
    h, w = binary.shape
    margin = 10
    for r in measure.regionprops(labels):
        if r.area < area_min or r.area > area_max:
            continue
        minr, minc, maxr, maxc = r.bbox
        if minr < margin or minc < margin or maxr > h - margin or maxc > w - margin:
            continue
        out[labels == r.label] = 255
    return out, threshold


# ---------------------------- stage 4 ---------------------------- #
def save_overlay(img_bgr: np.ndarray, mask: np.ndarray, out_path: Path) -> int:
    overlay = img_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 80), 2)
    cv2.imwrite(str(out_path), overlay)
    return len(contours)


def count_cells(mask: np.ndarray):
    labels = measure.label(mask > 0)
    return measure.regionprops(labels)


def classify_cell_type(region) -> str:
    area = float(region.area)
    perimeter = float(region.perimeter) if region.perimeter > 0 else 1.0
    circularity = float(4.0 * np.pi * area / ((perimeter**2) + 1e-6))

    if circularity > 0.70 and area < 800:
        return "leukocyte"
    if circularity < 0.55 or area > 2000:
        return "cornified"
    return "epithelial"


def list_images(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def main(args: argparse.Namespace) -> None:
    img_dir = Path(args.img_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    pseudo_dir = out_dir / "pseudo_masks"
    refined_dir = out_dir / "stage2_masks"
    overlay_dir = out_dir / "overlays"

    for d in (pseudo_dir, refined_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)
    stage_rules = load_stage_rules(args.stage_rules)

    img_paths = [p.resolve() for p in list_images(img_dir)]
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]
    print(f"Found {len(img_paths)} image(s).")

    # Stage 1
    print("\n-- STAGE 1: pseudo-mask generation --")
    pairs_stage1: list[tuple[Path, Path]] = []
    for p in tqdm(img_paths, desc="Pseudo-mask"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        mask = generate_pseudo_mask(
            img_bgr=img,
            area_min=args.area_min,
            area_max=args.area_max,
            circularity_min=args.circularity_min,
        )
        mp = pseudo_dir / f"{p.stem}_mask.png"
        cv2.imwrite(str(mp), mask)
        pairs_stage1.append((p, mp))

    if args.skip_unet:
        print("\n-- skip_unet: using Stage 1 pseudo-masks as final output --")
        rows: list[dict[str, object]] = []
        per_cell_rows: list[dict[str, object]] = []
        for p, mp in tqdm(pairs_stage1, desc="Stage1->Final"):
            img = cv2.imread(str(p))
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            n_cells = save_overlay(img_bgr=img, mask=mask, out_path=overlay_dir / f"{p.stem}_overlay.jpg")
            regions = count_cells(mask)
            areas = [r.area for r in regions]
            cell_counts = {"cornified": 0, "epithelial": 0, "leukocyte": 0}
            for idx, r in enumerate(regions):
                cell_type = classify_cell_type(r)
                cell_counts[cell_type] += 1
                perimeter = float(r.perimeter) if r.perimeter > 0 else 1.0
                circularity = float(4.0 * np.pi * float(r.area) / ((perimeter**2) + 1e-6))
                cy, cx = r.centroid
                minr, minc, maxr, maxc = r.bbox
                per_cell_rows.append(
                    {
                        "image": p.name,
                        "cell_id": idx,
                        "cell_type": cell_type,
                        "area_px": int(r.area),
                        "circularity": round(circularity, 4),
                        "centroid_x": round(float(cx), 2),
                        "centroid_y": round(float(cy), 2),
                        "bbox_minr": int(minr),
                        "bbox_minc": int(minc),
                        "bbox_maxr": int(maxr),
                        "bbox_maxc": int(maxc),
                    }
                )

            stage_input = {
                "Cornified epithelial cell": cell_counts["cornified"],
                "Nucleated epithelial cell": cell_counts["epithelial"],
                "Leukocyte": cell_counts["leukocyte"],
            }
            stage_result = infer_stage(stage_input, rules=stage_rules)
            rows.append(
                {
                    "image": p.name,
                    "n_cells": n_cells,
                    "cornified_count": cell_counts["cornified"],
                    "epithelial_count": cell_counts["epithelial"],
                    "leukocyte_count": cell_counts["leukocyte"],
                    "cornified_ratio": round(float(stage_result["cornified_ratio"]), 4),
                    "epithelial_ratio": round(float(stage_result["epithelial_ratio"]), 4),
                    "leukocyte_ratio": round(float(stage_result["leukocyte_ratio"]), 4),
                    "predicted_stage": str(stage_result["stage"]),
                    "mean_area_px": round(float(np.mean(areas)), 1) if areas else 0.0,
                    "std_area_px": round(float(np.std(areas)), 1) if areas else 0.0,
                    "min_area_px": int(min(areas)) if areas else 0,
                    "max_area_px": int(max(areas)) if areas else 0,
                    "threshold_used": "stage1_mask",
                }
            )

        df = pd.DataFrame(rows)
        per_image_csv = out_dir / "per_image_stage_summary.csv"
        legacy_csv = out_dir / "cell_statistics.csv"
        per_cell_csv = out_dir / "cell_level_features.csv"
        df.to_csv(per_image_csv, index=False)
        df.to_csv(legacy_csv, index=False)
        pd.DataFrame(per_cell_rows).to_csv(per_cell_csv, index=False)
        print(f"\nDone. Results: {out_dir}")
        print(f"Overlays: {overlay_dir}")
        print(f"Per-image CSV: {per_image_csv}")
        print(f"Per-cell CSV: {per_cell_csv}")
        print(f"Legacy stats CSV: {legacy_csv}")
        print(df.to_string(index=False))
        return

    # Stage 2
    print("\n-- STAGE 2: train Mini U-Net on pseudo-masks --")
    model = MiniUNet(base=args.base_channels).to(DEVICE)
    model = train_model(
        model=model,
        pairs=pairs_stage1,
        epochs=args.epochs_stage1,
        desc="Stage-1",
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        patches_per_img=args.patches_per_img,
        lr=args.lr,
    )

    # Stage 3
    print("\n-- STAGE 3: self-training refinement --")
    pairs_stage2: list[tuple[Path, Path]] = []
    stage3_thresholds: list[float] = []
    stage1_mask_by_name = {img_path.name: mask_path for img_path, mask_path in pairs_stage1}
    for p in tqdm(img_paths, desc="Re-infer"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        roi_mask = None
        if not args.disable_roi_only:
            mpath = stage1_mask_by_name.get(p.name)
            if mpath is not None:
                roi_mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)

        prob = infer_full_image(
            model=model,
            img_bgr=img,
            patch_size=args.patch_size,
            stride=args.stride,
            roi_mask=roi_mask,
            roi_dilate=args.roi_dilate,
        )
        refined, thr = mask_from_prob(
            prob=prob,
            conf_floor=args.conf_thresh,
            adaptive_top_percent=args.adaptive_top_percent,
            area_min=args.area_min,
            area_max=args.area_max,
        )
        stage3_thresholds.append(thr)
        mp = refined_dir / f"{p.stem}_mask2.png"
        cv2.imwrite(str(mp), refined)
        pairs_stage2.append((p, mp))
    if stage3_thresholds:
        print(
            f"Stage-3 adaptive threshold mean={np.mean(stage3_thresholds):.3f}, "
            f"min={np.min(stage3_thresholds):.3f}, max={np.max(stage3_thresholds):.3f}"
        )

    model = train_model(
        model=model,
        pairs=pairs_stage2,
        epochs=args.epochs_stage2,
        desc="Stage-2",
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        patches_per_img=args.patches_per_img,
        lr=args.lr,
    )
    torch.save(model.state_dict(), str(out_dir / "miniunet_bootstrap.pth"))
    print(f"Saved model: {out_dir / 'miniunet_bootstrap.pth'}")

    # Stage 4
    print("\n-- STAGE 4: final inference + overlays + stats --")
    rows: list[dict[str, object]] = []
    per_cell_rows: list[dict[str, object]] = []
    final_thresholds: list[float] = []
    stage2_mask_by_name = {img_path.name: mask_path for img_path, mask_path in pairs_stage2}
    for p in tqdm(img_paths, desc="Final"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        roi_mask = None
        if not args.disable_roi_only:
            mpath = stage2_mask_by_name.get(p.name)
            if mpath is not None:
                roi_mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)

        prob = infer_full_image(
            model=model,
            img_bgr=img,
            patch_size=args.patch_size,
            stride=args.stride,
            roi_mask=roi_mask,
            roi_dilate=args.roi_dilate,
        )
        final, thr = mask_from_prob(
            prob=prob,
            conf_floor=args.conf_thresh,
            adaptive_top_percent=args.adaptive_top_percent,
            area_min=args.area_min,
            area_max=args.area_max,
        )
        final_thresholds.append(thr)

        n_cells = save_overlay(img_bgr=img, mask=final, out_path=overlay_dir / f"{p.stem}_overlay.jpg")
        regions = count_cells(final)
        areas = [r.area for r in regions]
        cell_counts = {"cornified": 0, "epithelial": 0, "leukocyte": 0}
        for idx, r in enumerate(regions):
            cell_type = classify_cell_type(r)
            cell_counts[cell_type] += 1
            perimeter = float(r.perimeter) if r.perimeter > 0 else 1.0
            circularity = float(4.0 * np.pi * float(r.area) / ((perimeter**2) + 1e-6))
            cy, cx = r.centroid
            minr, minc, maxr, maxc = r.bbox
            per_cell_rows.append(
                {
                    "image": p.name,
                    "cell_id": idx,
                    "cell_type": cell_type,
                    "area_px": int(r.area),
                    "circularity": round(circularity, 4),
                    "centroid_x": round(float(cx), 2),
                    "centroid_y": round(float(cy), 2),
                    "bbox_minr": int(minr),
                    "bbox_minc": int(minc),
                    "bbox_maxr": int(maxr),
                    "bbox_maxc": int(maxc),
                }
            )

        stage_input = {
            "Cornified epithelial cell": cell_counts["cornified"],
            "Nucleated epithelial cell": cell_counts["epithelial"],
            "Leukocyte": cell_counts["leukocyte"],
        }
        stage_result = infer_stage(stage_input, rules=stage_rules)
        rows.append(
            {
                "image": p.name,
                "n_cells": n_cells,
                "cornified_count": cell_counts["cornified"],
                "epithelial_count": cell_counts["epithelial"],
                "leukocyte_count": cell_counts["leukocyte"],
                "cornified_ratio": round(float(stage_result["cornified_ratio"]), 4),
                "epithelial_ratio": round(float(stage_result["epithelial_ratio"]), 4),
                "leukocyte_ratio": round(float(stage_result["leukocyte_ratio"]), 4),
                "predicted_stage": str(stage_result["stage"]),
                "mean_area_px": round(float(np.mean(areas)), 1) if areas else 0.0,
                "std_area_px": round(float(np.std(areas)), 1) if areas else 0.0,
                "min_area_px": int(min(areas)) if areas else 0,
                "max_area_px": int(max(areas)) if areas else 0,
                "threshold_used": round(float(thr), 4),
            }
        )

    df = pd.DataFrame(rows)
    per_image_csv = out_dir / "per_image_stage_summary.csv"
    legacy_csv = out_dir / "cell_statistics.csv"
    per_cell_csv = out_dir / "cell_level_features.csv"
    df.to_csv(per_image_csv, index=False)
    df.to_csv(legacy_csv, index=False)
    pd.DataFrame(per_cell_rows).to_csv(per_cell_csv, index=False)
    print(f"\nDone. Results: {out_dir}")
    print(f"Overlays: {overlay_dir}")
    print(f"Per-image CSV: {per_image_csv}")
    print(f"Per-cell CSV: {per_cell_csv}")
    print(f"Legacy stats CSV: {legacy_csv}")
    if final_thresholds:
        print(
            f"Final adaptive threshold mean={np.mean(final_thresholds):.3f}, "
            f"min={np.min(final_thresholds):.3f}, max={np.max(final_thresholds):.3f}"
        )
    print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pseudo-label bootstrap cell segmentation (CPU)")
    parser.add_argument("--img_dir", default="data/raw", help="Folder with microscopy images")
    parser.add_argument("--out_dir", default="results/bootstrap", help="Output folder")
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Process only first N images (0 means all).",
    )
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patches_per_img", type=int, default=PATCHES_PER_IMG)
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH, help="Minimum confidence threshold floor.")
    parser.add_argument(
        "--adaptive_top_percent",
        type=float,
        default=ADAPTIVE_TOP_PERCENT,
        help="Adaptive top-K percentile for pseudo-label thresholding (0 disables adaptive mode).",
    )
    parser.add_argument("--area_min", type=int, default=AREA_MIN)
    parser.add_argument("--area_max", type=int, default=AREA_MAX)
    parser.add_argument("--circularity_min", type=float, default=CIRCULARITY_MIN)
    parser.add_argument("--epochs_stage1", type=int, default=EPOCHS_STAGE1)
    parser.add_argument("--epochs_stage2", type=int, default=EPOCHS_STAGE2)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--base_channels", type=int, default=BASE_CHANNELS, help="MiniUNet base channels.")
    parser.add_argument("--stride", type=int, default=48, help="Sliding-window stride.")
    parser.add_argument("--roi_dilate", type=int, default=11, help="ROI mask dilation size for ROI-only inference.")
    parser.add_argument("--stage_rules", default="config/stage_rules.yaml", help="YAML file for stage inference rules.")
    parser.add_argument(
        "--skip_unet",
        action="store_true",
        help="Skip U-Net training and use Stage 1 pseudo-masks as final output.",
    )
    parser.add_argument(
        "--disable_roi_only",
        action="store_true",
        help="Disable ROI-only inference and force full-image sliding window.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

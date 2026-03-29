#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

try:
    import scipy.io
    HAS_SCIPY_IO = True
except Exception:
    HAS_SCIPY_IO = False

try:
    from torchvision.transforms import InterpolationMode
    TV_BICUBIC = InterpolationMode.BICUBIC
except Exception:
    TV_BICUBIC = None

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)
ANN_EXTS = [".mat", ".xml"]
NEIGH_8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def list_images_recursive(root: str, max_images_per_dir: int = 0) -> List[str]:
    out = []
    for dp, _, files in os.walk(root):
        imgs = []
        for f in files:
            if Path(f).suffix in EXTS:
                imgs.append(str(Path(dp) / f))
        imgs.sort()
        if max_images_per_dir and max_images_per_dir > 0:
            imgs = imgs[:max_images_per_dir]
        out.extend(imgs)
    out.sort()
    return out


def infer_category_from_dirname(dirname: str) -> str:
    return dirname.split("_")[0]


def parse_voc_xml_boxes(xml_path: Path) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    root = ET.parse(str(xml_path)).getroot()
    out = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        cls = (name_el.text or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue

        def getv(tag: str) -> int:
            t = bb.find(tag)
            return int(float(t.text)) if (t is not None and t.text is not None) else 0

        xmin = getv("xmin")
        ymin = getv("ymin")
        xmax = getv("xmax")
        ymax = getv("ymax")
        x1 = max(0, xmin - 1)
        y1 = max(0, ymin - 1)
        x2 = max(x1 + 1, xmax)
        y2 = max(y1 + 1, ymax)
        out.append((cls, (x1, y1, x2, y2)))
    return out


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return list(x.reshape(-1))
    return [x]


def get_field(obj: Any, key: str) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return None


def parse_pascal3d_mat_boxes(mat_path: Path) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    if not HAS_SCIPY_IO:
        raise RuntimeError("scipy.io not available")
    md = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    record = md.get("record", None)
    if record is None:
        record = md.get("annotation", None)
    if record is None:
        for k in ["rec", "ann", "data"]:
            if k in md:
                record = md[k]
                break
    if record is None:
        raise RuntimeError("No record/annotation key found")

    objects = get_field(record, "objects")
    if objects is None:
        objects = get_field(record, "object")
    objs = as_list(objects)

    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for o in objs:
        cls = get_field(o, "class")
        if cls is None:
            cls = get_field(o, "name")
        if cls is None:
            cls = get_field(o, "classname")
        if cls is None:
            continue
        cls = str(cls).strip()

        bbox = get_field(o, "bbox")
        if bbox is None:
            bbox = get_field(o, "bndbox")
        if bbox is None:
            continue

        bb = np.asarray(bbox).reshape(-1)
        if bb.size < 4:
            continue
        xmin, ymin, xmax, ymax = [float(bb[i]) for i in range(4)]
        x1 = max(0, int(round(xmin)) - 1)
        y1 = max(0, int(round(ymin)) - 1)
        x2 = max(x1 + 1, int(round(xmax)))
        y2 = max(y1 + 1, int(round(ymax)))
        out.append((cls, (x1, y1, x2, y2)))
    return out


def parse_boxes(ann_path: Path) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    suf = ann_path.suffix.lower()
    if suf == ".xml":
        return parse_voc_xml_boxes(ann_path)
    if suf == ".mat":
        return parse_pascal3d_mat_boxes(ann_path)
    raise RuntimeError(f"Unsupported annotation ext: {suf}")


def index_annotation_files(ann_root: Path) -> Dict[str, List[Path]]:
    m: Dict[str, List[Path]] = {}
    for ext in ANN_EXTS:
        for p in ann_root.rglob(f"*{ext}"):
            m.setdefault(p.stem, []).append(p)
    return m


def mask_to_bbox(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2, y2)


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def center_offset_norm(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], h: int, w: int) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    diag = math.hypot(w, h)
    return float(math.hypot(acx - bcx, acy - bcy) / max(1e-6, diag))


def preprocess_short_resize_pad(
    img_pil: Image.Image,
    short_side: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    patch: int,
    device: str,
    keep_ar: bool = False,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    ho, wo = img_pil.size[1], img_pil.size[0]
    if TV_BICUBIC is None:
        resize_op = T.Resize(short_side) if keep_ar else T.Resize((short_side, short_side))
    else:
        resize_op = T.Resize(short_side, interpolation=TV_BICUBIC) if keep_ar else T.Resize((short_side, short_side), interpolation=TV_BICUBIC)

    tfm = T.Compose([T.ToTensor(), resize_op, T.Normalize(mean, std)])
    x = tfm(img_pil)
    hr, wr = int(x.shape[-2]), int(x.shape[-1])

    hp = int(math.ceil(hr / patch) * patch)
    wp = int(math.ceil(wr / patch) * patch)

    if hp != hr or wp != wr:
        pad = torch.zeros((3, hp, wp), dtype=x.dtype)
        pad[:, :hr, :wr] = x
        x = pad

    ten = x.unsqueeze(0).to(device=device, dtype=torch.float32)
    return ten, (hp, wp), (hr, wr), (ho, wo)


def pil_to_rgb_float(img_pil: Image.Image) -> np.ndarray:
    return np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0


def load_dino_vits16(device: str):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model.eval().to(device)
    return model


def load_dinov2_vits14(device: str):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model


@torch.no_grad()
def tokencut_patch_feats_dino(model, img_pil: Image.Image, device: str, resize_short: int, patch: int = 16, keep_ar: bool = False):
    ten, pad_hw, resized_hw, orig_hw = preprocess_short_resize_pad(
        img_pil, short_side=resize_short, mean=IMAGENET_MEAN, std=IMAGENET_STD, patch=patch, device=device, keep_ar=keep_ar
    )
    feats = model.get_intermediate_layers(ten, n=1)[0][:, 1:, :]
    feats = F.normalize(feats, dim=-1)
    return feats.squeeze(0), pad_hw, resized_hw, orig_hw, patch


def build_affinity(features: torch.Tensor, tau: float = 0.2, eps: float = 1e-5) -> csr_matrix:
    x = features.detach().cpu().numpy().astype(np.float32)
    sim = (x @ x.T).astype(np.float32)
    a = (sim >= tau).astype(np.float32)
    np.fill_diagonal(a, 1.0)
    a[a == 0] = eps
    return csr_matrix(a)


def normalized_cut_second_eigvec(a: csr_matrix) -> np.ndarray:
    d = np.asarray(a.sum(axis=1)).reshape(-1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    dm = csr_matrix(np.diag(dinv))
    i = csr_matrix(np.eye(a.shape[0], dtype=np.float32))
    lsym = i - dm @ a @ dm
    vals, vecs = eigsh(lsym, k=2, which="SM")
    idx = np.argsort(vals)
    y = dinv * vecs[:, idx[1]]
    return y


def bipartition_to_fg(y: np.ndarray) -> np.ndarray:
    thr = float(y.mean())
    a = y <= thr
    b = y > thr
    fg = b if np.mean(np.abs(y[b])) >= np.mean(np.abs(y[a])) else a
    return fg.astype(np.uint8)


def vecmask_to_full(mask_vec: np.ndarray, feat_hw, pad_hw, resized_hw, orig_hw) -> np.ndarray:
    hf, wf = feat_hw
    hp, wp = pad_hw
    hr, wr = resized_hw
    ho, wo = orig_hw
    grid = torch.from_numpy(mask_vec.reshape(hf, wf).astype(np.float32))[None, None]
    grid = F.interpolate(grid, size=(hp, wp), mode="nearest")
    grid = grid[:, :, :hr, :wr]
    grid = F.interpolate(grid, size=(ho, wo), mode="nearest")
    return (grid.squeeze().numpy() > 0.5).astype(np.uint8)


def _try_import_bilateral_solver():
    here = Path(__file__).resolve()
    proj = here.parents[1]
    cand_paths = [
        proj / "tokencut_bs",
        proj / "TokenCut",
        proj / "external" / "tokencut_bs",
        proj / "external" / "TokenCut",
    ]
    for p in cand_paths:
        if p.exists():
            sys.path.insert(0, str(p))

    tried = []
    for modname in [
        "bilateral_solver",
        "bilateral_solver.bilateral_solver",
        "tokencut_bs.bilateral_solver",
        "tokencut_bs.bilateral_solver.bilateral_solver",
        "TokenCut.bilateral_solver",
        "TokenCut.bilateral_solver.bilateral_solver",
    ]:
        try:
            m = __import__(modname, fromlist=["*"])
            return m, tried
        except Exception as e:
            tried.append((modname, repr(e)))
    return None, tried


def patch_bilateral_solver_cg_compat(mod) -> None:
    try:
        import inspect
        import scipy.sparse.linalg as sla
    except Exception:
        return

    try:
        sig = inspect.signature(sla.cg)
        has_tol = "tol" in sig.parameters
    except Exception:
        has_tol = False

    if has_tol:
        return

    orig_cg = sla.cg

    def cg_compat(A, b, x0=None, tol=None, maxiter=None, M=None, callback=None, atol=0.0, **kwargs):
        if tol is not None:
            kwargs.setdefault("rtol", float(tol))
            kwargs.setdefault("atol", float(atol))
        return orig_cg(A, b, x0=x0, maxiter=maxiter, M=M, callback=callback, **kwargs)

    if hasattr(mod, "cg"):
        try:
            setattr(mod, "cg", cg_compat)
        except Exception:
            pass
    try:
        sla.cg = cg_compat
    except Exception:
        pass


def bilateral_refine_mask(
    img_rgb_float: np.ndarray,
    init_mask01: np.ndarray,
    sigma_spatial: float = 16.0,
    sigma_luma: float = 8.0,
    sigma_chroma: float = 8.0,
    lam: float = 128.0,
    cg_tol: float = 1e-5,
    cg_maxiter: int = 25,
    thresh: float = 0.5,
) -> np.ndarray:
    mod, tried = _try_import_bilateral_solver()
    if mod is None:
        msg = "[bilateral] import failed. Tried:\n" + "\n".join([f"  - {a}: {b}" for a, b in tried])
        raise RuntimeError(msg)

    patch_bilateral_solver_cg_compat(mod)

    bilateral_grid = getattr(mod, "BilateralGrid", None)
    bilateral_solver = getattr(mod, "BilateralSolver", None)
    if bilateral_grid is None or bilateral_solver is None:
        bilateral_grid = getattr(mod, "bilateral_grid", None)
        bilateral_solver = getattr(mod, "bilateral_solver", None)
    if bilateral_grid is None or bilateral_solver is None:
        raise RuntimeError("Could not find BilateralGrid/BilateralSolver")

    h, w = init_mask01.shape
    y = init_mask01.astype(np.float32).reshape(-1, 1)
    ww = np.ones_like(y, dtype=np.float32)

    img1 = img_rgb_float.astype(np.float32)
    img255 = (img1 * 255.0).astype(np.float32)
    params = {
        "lam": float(lam),
        "A_diag_min": 1e-5,
        "cg_tol": float(cg_tol),
        "cg_maxiter": int(cg_maxiter),
    }

    last_err = None
    for guide in [img1, img255]:
        try:
            grid = bilateral_grid(guide, sigma_spatial=sigma_spatial, sigma_luma=sigma_luma, sigma_chroma=sigma_chroma)
            solver = bilateral_solver(grid, params)
            out = solver.solve(y, ww)
            out = np.asarray(out).reshape(h, w)
            return (out >= thresh).astype(np.uint8)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"[bilateral] solve failed: {repr(last_err)}")


@torch.no_grad()
def dinov2_patch_tokens_shortpad(model, img_pil: Image.Image, device: str, resize_short: int, patch: int = 14, keep_ar: bool = False):
    ten, pad_hw, resized_hw, orig_hw = preprocess_short_resize_pad(
        img_pil, short_side=resize_short, mean=HALF_MEAN, std=HALF_STD, patch=patch, device=device, keep_ar=keep_ar
    )
    emb = model.forward_features(ten)
    x = emb["x_norm_patchtokens"].detach().cpu().numpy()[0].astype(np.float32)
    hp, wp = pad_hw
    ph, pw = hp // patch, wp // patch
    return x, resized_hw, pad_hw, orig_hw, (ph, pw)


def mask_to_patch_fg(mask_full: np.ndarray, resized_hw: Tuple[int, int], pad_hw: Tuple[int, int], patch: int) -> np.ndarray:
    hr, wr = resized_hw
    hp, wp = pad_hw
    ten = torch.from_numpy(mask_full.astype(np.float32))[None, None]
    ten = F.interpolate(ten, size=(hr, wr), mode="nearest")
    if hp != hr or wp != wr:
        pad = torch.zeros((1, 1, hp, wp), dtype=ten.dtype)
        pad[:, :, :hr, :wr] = ten
        ten = pad
    m = ten[0, 0].numpy()

    ph, pw = hp // patch, wp // patch
    patch_fg = np.zeros((ph, pw), dtype=bool)
    for r in range(ph):
        y0 = r * patch
        y1 = min((r + 1) * patch, hp)
        for c in range(pw):
            x0 = c * patch
            x1 = min((c + 1) * patch, wp)
            if y0 >= hr or x0 >= wr:
                continue
            if m[y0:y1, x0:x1].mean() > 0.0:
                patch_fg[r, c] = True
    return patch_fg


def connected_components_bool(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            st = [(r, c)]
            vis[r, c] = True
            comp = []
            while st:
                rr, cc = st.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGH_8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        st.append((nr, nc))
            comps.append(comp)
    return comps


def kmeans_label_map(feats: np.ndarray, patch_fg: np.ndarray, k: int, seed: int) -> np.ndarray:
    ph, pw = patch_fg.shape
    p = ph * pw
    label = -1 * np.ones((p,), np.int32)
    idx = np.where(patch_fg.reshape(-1))[0]
    if len(idx) > 0:
        k_eff = max(1, min(k, len(idx)))
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=seed)
        labs = km.fit_predict(feats[idx])
        label[idx] = labs
    return label.reshape(ph, pw)


def absorb_size1_noise(label_map: np.ndarray) -> np.ndarray:
    ph, pw = label_map.shape
    lab = label_map.copy()
    kmax = int(lab[lab >= 0].max()) if np.any(lab >= 0) else -1
    for cid in range(kmax + 1):
        comps = connected_components_bool(lab == cid)
        for comp in comps:
            if len(comp) != 1:
                continue
            r, c = comp[0]
            cnt = {}
            for dr, dc in NEIGH_8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < ph and 0 <= cc < pw:
                    v = int(lab[rr, cc])
                    if v < 0 or v == cid:
                        continue
                    cnt[v] = cnt.get(v, 0) + 1
            if cnt:
                new_lab = max(cnt.items(), key=lambda x: x[1])[0]
                lab[r, c] = new_lab
    return lab


def nodes_from_labelmap(label_map: np.ndarray, feats: np.ndarray, resized_hw: Tuple[int, int], patch: int):
    hr, wr = resized_hw
    ph, pw = label_map.shape
    nodes = []
    kmax = int(label_map[label_map >= 0].max()) if np.any(label_map >= 0) else -1
    nid = 0
    for cid in range(kmax + 1):
        comps = connected_components_bool(label_map == cid)
        for comp in comps:
            rr = np.array([p[0] for p in comp], float)
            cc = np.array([p[1] for p in comp], float)

            cy = (rr.mean() + 0.5) * patch
            cx = (cc.mean() + 0.5) * patch
            cy = float(np.clip(cy, 0.0, max(1.0, hr - 1.0)))
            cx = float(np.clip(cx, 0.0, max(1.0, wr - 1.0)))

            mask = np.zeros((ph, pw), bool)
            for r, c in comp:
                mask[r, c] = True
            idx = np.where(mask.reshape(-1))[0]
            feat_mean = feats[idx].mean(axis=0) if len(idx) > 0 else np.zeros((feats.shape[1],), np.float32)

            nodes.append(
                {
                    "id": int(nid),
                    "x_norm": float(cx / wr),
                    "y_norm": float(cy / hr),
                    "cluster": int(cid),
                    "size": int(len(comp)),
                    "feat": feat_mean.tolist(),
                }
            )
            nid += 1
    return nodes


def complete_graph_edges(nodes: List[Dict[str, Any]], h: int, w: int):
    diag = math.hypot(w, h)
    edges = []
    n = len(nodes)
    for i in range(n):
        xi = nodes[i]["x_norm"] * w
        yi = nodes[i]["y_norm"] * h
        for j in range(i + 1, n):
            xj = nodes[j]["x_norm"] * w
            yj = nodes[j]["y_norm"] * h
            d = math.hypot(xi - xj, yi - yj) / max(1e-6, diag)
            edges.append({"u": int(nodes[i]["id"]), "v": int(nodes[j]["id"]), "dist": float(d)})
    return edges


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ann-root", type=str, default="")
    ap.add_argument("--tokencut-backbone", choices=["dino"], default="dino")
    ap.add_argument("--tok-resize", type=int, default=480)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--fg-mode", choices=["tokencut", "full"], default="tokencut")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--k-list", type=int, nargs="+", default=[16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-images-per-dir", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    img_paths = list_images_recursive(str(data_root), max_images_per_dir=int(args.max_images_per_dir))
    if args.max_images and len(img_paths) > args.max_images:
        img_paths = img_paths[: args.max_images]
    if not img_paths:
        raise RuntimeError(f"no images under: {args.data_root}")

    ann_root = Path(args.ann_root) if args.ann_root else None
    ann_map: Dict[str, List[Path]] = {}
    if ann_root is not None:
        ann_map = index_annotation_files(ann_root)

    tok_dino = load_dino_vits16(device) if args.fg_mode == "tokencut" else None
    d2_graph = load_dinov2_vits14(device)

    ks = list(dict.fromkeys(args.k_list))
    per_k_summary: Dict[int, Dict[str, Any]] = {}
    graph_index: Dict[str, List[Dict[str, Any]]] = {}
    for k in ks:
        per_k_summary[k] = {
            "num_graphs_saved": 0,
            "num_graph_build_failed": 0,
        }
        graph_index[str(k)] = []

    ann_eval = {
        "enabled": ann_root is not None,
        "num_ok": 0,
        "num_error": 0,
        "ok_rows": [],
        "errors": [],
    }

    all_errors: List[Dict[str, Any]] = []
    total_start = time.time()

    for ip, p in enumerate(img_paths, 1):
        img_path = Path(p)
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            all_errors.append({"image": str(img_path), "stage": "open_image", "detail": repr(e)})
            for k in ks:
                per_k_summary[k]["num_graph_build_failed"] += 1
            continue

        ho, wo = img.size[1], img.size[0]

        try:
            if args.fg_mode == "full":
                fg_full = np.ones((ho, wo), dtype=np.uint8)
                fg_info = {"mode": "full", "fg_area_ratio": float(fg_full.mean())}
            else:
                feats_tok, (hp, wp), (hr, wr), (ho2, wo2), patch_tok = tokencut_patch_feats_dino(
                    tok_dino, img, device, resize_short=args.tok_resize, patch=16, keep_ar=False
                )
                a = build_affinity(feats_tok, tau=args.tau, eps=args.eps)
                y = normalized_cut_second_eigvec(a)
                fg_vec = bipartition_to_fg(y)
                hf = hp // patch_tok
                wf = wp // patch_tok
                fg_init = vecmask_to_full(fg_vec, (hf, wf), (hp, wp), (hr, wr), (ho2, wo2))
                img_rgb_float = pil_to_rgb_float(img)
                fg_full = bilateral_refine_mask(img_rgb_float, fg_init)
                fg_info = {
                    "mode": "tokencut",
                    "fg_area_ratio": float(fg_full.mean()),
                    "tokencut_patch_grid_hw": [int(hf), int(wf)],
                }
        except Exception as e:
            all_errors.append({"image": str(img_path), "stage": "foreground", "detail": repr(e)})
            for k in ks:
                per_k_summary[k]["num_graph_build_failed"] += 1
            continue

        if ann_root is not None:
            stem = img_path.stem
            target_class = infer_category_from_dirname(img_path.parent.name)
            cands = ann_map.get(stem, [])
            if not cands:
                ann_eval["num_error"] += 1
                ann_eval["errors"].append(
                    {
                        "image": str(img_path),
                        "reason": "annotation_not_found",
                        "detail": f"stem={stem} target_class={target_class}",
                    }
                )
            else:
                ann_path = cands[0]
                try:
                    boxes = parse_boxes(ann_path)
                    match = [b for (cls, b) in boxes if cls == target_class]
                    if len(match) == 0:
                        ann_eval["num_error"] += 1
                        ann_eval["errors"].append(
                            {
                                "image": str(img_path),
                                "reason": "no_bbox_for_target_class",
                                "detail": f"target={target_class} ann={ann_path}",
                            }
                        )
                    else:
                        pred_box = mask_to_bbox(fg_full)
                        max_iou = 0.0
                        best_gt = None
                        if pred_box is not None:
                            for gb in match:
                                v = iou_xyxy(pred_box, gb)
                                if v > max_iou:
                                    max_iou = v
                                    best_gt = gb
                        offset = 1.0 if (pred_box is None or best_gt is None) else center_offset_norm(pred_box, best_gt, ho, wo)
                        ann_eval["num_ok"] += 1
                        ann_eval["ok_rows"].append(
                            {
                                "image": str(img_path),
                                "target_class": target_class,
                                "ann_path": str(ann_path),
                                "num_match_boxes": int(len(match)),
                                "max_iou": float(max_iou),
                                "center_offset": float(offset),
                                "fg_area_ratio": float(fg_full.mean()),
                            }
                        )
                except Exception as e:
                    ann_eval["num_error"] += 1
                    ann_eval["errors"].append(
                        {
                            "image": str(img_path),
                            "reason": "annotation_parse_failed",
                            "detail": f"ann={ann_path} err={repr(e)}",
                        }
                    )

        try:
            feats14, resized_hw, pad_hw, _, _ = dinov2_patch_tokens_shortpad(
                d2_graph, img, device=device, resize_short=args.img_size, patch=14, keep_ar=False
            )
            patch_fg = mask_to_patch_fg(fg_full, resized_hw=resized_hw, pad_hw=pad_hw, patch=14)
            rel = img_path.relative_to(data_root)
            stem2 = rel.stem
            sub = rel.parent
            hr2, wr2 = resized_hw

            for k in ks:
                label0 = kmeans_label_map(feats14, patch_fg, k=k, seed=args.seed)
                label = absorb_size1_noise(label0)
                nodes = nodes_from_labelmap(label, feats14, resized_hw=resized_hw, patch=14)
                edges = complete_graph_edges(nodes, h=hr2, w=wr2)

                out_graph_path = out_root / sub / f"{stem2}__k{k}.json"
                save_json(
                    out_graph_path,
                    {
                        "image_path": str(img_path),
                        "relative_image_path": str(rel),
                        "object_name": infer_category_from_dirname(img_path.parent.name),
                        "direction_name": "_".join(img_path.parent.name.split("_")[1:]) if "_" in img_path.parent.name else "unknown",
                        "H": int(hr2),
                        "W": int(wr2),
                        "k": int(k),
                        "foreground": fg_info,
                        "nodes": nodes,
                        "edges": edges,
                    },
                )

                edge_dists = [float(e["dist"]) for e in edges]
                per_k_summary[k]["num_graphs_saved"] += 1
                graph_index[str(k)].append(
                    {
                        "image_path": str(img_path),
                        "relative_image_path": str(rel),
                        "graph_json_path": str(out_graph_path),
                        "num_nodes": int(len(nodes)),
                        "num_edges": int(len(edges)),
                        "foreground_area_ratio": float(fg_info["fg_area_ratio"]),
                        "mean_edge_dist": float(np.mean(edge_dists)) if edge_dists else None,
                    }
                )

        except Exception as e:
            all_errors.append({"image": str(img_path), "stage": "graph_build", "detail": repr(e)})
            for k in ks:
                per_k_summary[k]["num_graph_build_failed"] += 1

        if ip % 50 == 0 or ip == len(img_paths):
            print(f"[graphs] {ip}/{len(img_paths)}")

    ann_summary = None
    if ann_root is not None:
        corlocs = []
        areas = []
        offsets = []
        for row in ann_eval["ok_rows"]:
            corlocs.append(1.0 if row["max_iou"] >= 0.5 else 0.0)
            areas.append(float(row["fg_area_ratio"]))
            offsets.append(float(row["center_offset"]))
        ann_summary = {
            "ann_root": str(ann_root),
            "num_ok": int(ann_eval["num_ok"]),
            "num_error": int(ann_eval["num_error"]),
            "corloc_mean_at_iou_0_5": float(np.mean(corlocs)) if corlocs else None,
            "fg_area_ratio_mean": float(np.mean(areas)) if areas else None,
            "center_offset_mean": float(np.mean(offsets)) if offsets else None,
            "errors": ann_eval["errors"],
        }

    summary = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "device": device,
        "num_images_found": int(len(img_paths)),
        "seed": int(args.seed),
        "settings": {
            "tokencut_backbone": str(args.tokencut_backbone),
            "tok_resize": int(args.tok_resize),
            "tau": float(args.tau),
            "eps": float(args.eps),
            "fg_mode": str(args.fg_mode),
            "img_size": int(args.img_size),
            "k_list": [int(k) for k in ks],
            "max_images": int(args.max_images),
            "max_images_per_dir": int(args.max_images_per_dir),
        },
        "per_k": {str(k): per_k_summary[k] for k in ks},
        "annotation_eval": ann_summary,
        "errors": all_errors,
        "elapsed_sec": float(time.time() - total_start),
    }

    save_json(out_root / "graph_build_summary.json", summary)
    save_json(out_root / "graph_index.json", {"per_k": graph_index})

    total_saved = sum(v["num_graphs_saved"] for v in per_k_summary.values())
    if total_saved == 0:
        raise RuntimeError("No graphs were saved. See graph_build_summary.json for details.")

    print(f"[DONE] elapsed_sec={summary['elapsed_sec']:.1f}")


if __name__ == "__main__":
    main()
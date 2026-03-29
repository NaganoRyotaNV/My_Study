#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    class _NullTqdm:
        def __init__(self, iterable=None, total=None, desc=None, leave=False, dynamic_ncols=True):
            self.iterable = iterable
            self.total = total
            self.desc = desc

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n=1):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def set_description(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    def tqdm(iterable=None, *args, **kwargs):
        return _NullTqdm(iterable=iterable, **kwargs)


DIRS_ORDER_5 = ["front", "front-oblique", "side", "back-oblique", "back"]
DIRS_ORDER_9 = [
    "front",
    "front/front-oblique",
    "front-oblique",
    "front-oblique/side",
    "side",
    "side/back-oblique",
    "back-oblique",
    "back-oblique/back",
    "back",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def patch_scipy_for_pot() -> None:
    try:
        import importlib
        ls = importlib.import_module("scipy.optimize.linesearch")
        if not hasattr(ls, "scalar_search_armijo"):
            def scalar_search_armijo(phi, derphi, phi0, old_phi0=None, args=(), c1=1e-4, alpha0=1.0, amin=0.0):
                phi_a0 = phi(alpha0, *args)
                derphi_a0 = derphi(alpha0, *args)
                return alpha0, phi_a0, derphi_a0
            ls.scalar_search_armijo = scalar_search_armijo
    except Exception:
        pass


def import_ot():
    try:
        import ot
        return ot
    except Exception:
        patch_scipy_for_pot()
        import ot
        return ot


def try_import_gromov_fn(name: str):
    try:
        import ot.gromov as g
        return getattr(g, name)
    except Exception:
        return None


def call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**ok)


def parse_obj_dir_from_parent(path: Path) -> Tuple[str, str]:
    cand = path.parent.name
    if "_" in cand:
        obj = cand.split("_")[0]
        direction = "_".join(cand.split("_")[1:])
        return obj, direction
    return cand, "unknown"


def uniform_p(n: int) -> np.ndarray:
    p = np.ones((n,), dtype=np.float64)
    return p / max(p.sum(), 1e-12)


def build_structure_cost(n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    if n <= 0 or not edges:
        return np.zeros((n, n), dtype=np.float64)
    max_d = max(float(d) for _, _, d in edges)
    max_d = max(max_d, 1e-6)
    c = np.full((n, n), max_d, dtype=np.float64)
    np.fill_diagonal(c, 0.0)
    for u, v, d in edges:
        if d < c[u, v]:
            c[u, v] = d
            c[v, u] = d
    c = 0.5 * (c + c.T)
    np.fill_diagonal(c, 0.0)
    return c


def load_graph_json(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float]], str, str]:
    j = read_json(path)
    nodes = j.get("nodes", [])
    edges = j.get("edges", [])
    if not nodes:
        raise ValueError(f"no nodes: {path}")
    feat = np.stack([np.asarray(n["feat"], np.float64) for n in nodes], axis=0)
    coord = np.stack(
        [np.array([float(n.get("x_norm", 0.0)), float(n.get("y_norm", 0.0))], np.float64) for n in nodes],
        axis=0,
    )
    e_list = []
    for e in edges:
        u = int(e["u"])
        v = int(e["v"])
        d = float(e.get("dist", 1.0))
        if u > v:
            u, v = v, u
        e_list.append((u, v, d))
    obj, direction = parse_obj_dir_from_parent(path)
    return feat, coord, e_list, obj, direction


def dist_balanced(ot, fgw2, x1, c1, p, x2, c2, q, alpha: float) -> float:
    m = ot.dist(x1, x2, metric="euclidean") ** 2
    return float(fgw2(m, c1, c2, p, q, loss_fun="square_loss", alpha=float(alpha)))


def dist_partial(ot, pfgw2, x1, c1, p, x2, c2, q, alpha: float, mass: float) -> float:
    m = ot.dist(x1, x2, metric="euclidean") ** 2
    d = call_with_supported_kwargs(
        pfgw2,
        M=m,
        C1=c1,
        C2=c2,
        p=p,
        q=q,
        m=float(mass),
        loss_fun="square_loss",
        alpha=float(alpha),
        log=False,
    )
    return float(d)


def coupling_balanced(ot, fgw, x1, c1, p, x2, c2, q, alpha: float) -> np.ndarray:
    m = ot.dist(x1, x2, metric="euclidean") ** 2
    return np.asarray(fgw(m, c1, c2, p, q, loss_fun="square_loss", alpha=float(alpha), log=False), dtype=np.float64)


def coupling_partial(ot, pfgw, x1, c1, p, x2, c2, q, alpha: float, mass: float) -> np.ndarray:
    m = ot.dist(x1, x2, metric="euclidean") ** 2
    return np.asarray(
        call_with_supported_kwargs(
            pfgw,
            M=m,
            C1=c1,
            C2=c2,
            p=p,
            q=q,
            m=float(mass),
            loss_fun="square_loss",
            alpha=float(alpha),
            log=False,
        ),
        dtype=np.float64,
    )


def softmax_from_negdist(dists: np.ndarray, temp: float) -> np.ndarray:
    x = -np.asarray(dists, dtype=np.float64) / max(float(temp), 1e-12)
    finite = np.isfinite(x)
    if np.any(finite):
        x = x - np.max(x[finite])
    ex = np.exp(np.clip(x, -60.0, 60.0))
    ex[~np.isfinite(ex)] = 0.0
    s = float(ex.sum())
    if s <= 0.0:
        return np.ones_like(ex) / max(1, ex.size)
    return ex / s


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def plot_confusion(cm: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    cm = cm.astype(np.float64)
    row = cm.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    cmn = cm / row

    plt.figure(figsize=(8, 6))
    plt.imshow(cmn, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_legacy(cm: np.ndarray, labels: List[str], out_png: Path, title: str, normalize: bool) -> None:
    a = np.asarray(cm, dtype=np.float64)
    if normalize:
        row = a.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        a = a / row

    plt.figure(figsize=(8, 6))
    plt.imshow(a, interpolation="nearest")
    plt.title(title + (" (norm)" if normalize else ""))
    plt.colorbar(fraction=0.046, pad=0.04)
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=90, fontsize=8)
    plt.yticks(tick, labels, fontsize=8)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_grid_by_object(
    per_obj_cm: Dict[str, np.ndarray],
    labels: List[str],
    out_png: Path,
    title: str,
    ncols: int = 4,
) -> None:
    objs = sorted(per_obj_cm.keys())
    if not objs:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        ax.set_title(title)
        ensure_dir(out_png.parent)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(objs)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig_w = max(14, 4.0 * ncols)
    fig_h = max(9, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    last_im = None
    tick = np.arange(len(labels))
    for idx, obj in enumerate(objs):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        cm = np.asarray(per_obj_cm[obj], dtype=np.float64)
        row = cm.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        cmn = cm / row
        last_im = ax.imshow(cmn, interpolation="nearest", vmin=0.0, vmax=1.0)
        acc = float(np.trace(cm) / max(1.0, cm.sum()))
        ax.set_title(f"{obj}\nacc={acc:.3f}", fontsize=10)
        ax.set_xticks(tick)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick)
        ax.set_yticklabels(labels, fontsize=8)
        if c != 0:
            ax.set_yticklabels([])
        if r != nrows - 1:
            ax.set_xticklabels([])

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=14)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.ax.set_ylabel("row-normalized", rotation=90)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def list_jsons(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.json"))


def infer_run_paths(run_dir: Path) -> Dict[str, Path]:
    cfg = load_yaml(run_dir / "config_effective.yaml")
    k = int(cfg.get("graph", {}).get("k", 16))

    graphs_candidates = [
        run_dir / "graphs",
        run_dir / "graphs" / f"k{k}" / "graphs",
    ]
    graphs_root = next((p for p in graphs_candidates if p.is_dir()), graphs_candidates[0])

    return {
        "graphs_root": graphs_root,
        "reps_root": run_dir / "reps",
        "megagraph_root": run_dir / "megagraphs",
        "inference_json": run_dir / "inference" / "inference_result.json",
        "config": run_dir / "config_effective.yaml",
        "k": Path(str(k)),
    }


def collect_structure_train_templates(graphs_root: Path):
    train_paths = sorted((graphs_root / "train").rglob("*.json"))
    train_graphs = []
    obj_set = set()
    for p in train_paths:
        try:
            feat, _coord, edges, obj, _dir = load_graph_json(p)
            c = build_structure_cost(feat.shape[0], edges)
            train_graphs.append((p, feat, c, uniform_p(feat.shape[0]), obj))
            obj_set.add(obj)
        except Exception:
            continue
    labels = sorted(obj_set)
    return train_graphs, labels


def discover_reps(rep_root: Path) -> Dict[str, List[Path]]:
    base = rep_root / "reps"
    mp: Dict[str, List[Path]] = {}
    if not base.is_dir():
        return mp
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        for ddir in sorted(obj_dir.iterdir()):
            p = ddir / "rep_k16.json"
            if p.is_file():
                mp.setdefault(obj, []).append(p)
    return mp


def collect_rep_templates(rep_root: Path):
    rep_map = discover_reps(rep_root)
    rep_cache: Dict[str, List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray, str]]] = {}
    labels = sorted(rep_map.keys())
    for obj in labels:
        items = []
        for p in rep_map[obj]:
            try:
                feat, _coord, edges, _o, _direction_unused = load_graph_json(p)
                c = build_structure_cost(feat.shape[0], edges)
                direction = p.parent.name
                items.append((p, feat, c, uniform_p(feat.shape[0]), direction))
            except Exception:
                continue
        rep_cache[obj] = items
    return rep_cache, labels


def load_megagraphs(mega_root: Path):
    base = mega_root / "megagraphs"
    mp = {}
    if not base.is_dir():
        return mp
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        p = obj_dir / "megagraph.json"
        if not p.is_file():
            continue
        try:
            feat, _coord, edges, _o, _d = load_graph_json(p)
            c = build_structure_cost(feat.shape[0], edges)
            mp[obj] = (p, feat, c, uniform_p(feat.shape[0]))
        except Exception:
            continue
    return mp


def load_megagraph_member_label9(mega_root: Path) -> Dict[str, Dict[int, str]]:
    base = mega_root / "megagraphs"
    out: Dict[str, Dict[int, str]] = {}
    if not base.is_dir():
        return out

    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        p = obj_dir / "megagraph_members.json"
        if not p.is_file():
            continue

        try:
            j = read_json(p)
        except Exception:
            continue

        node_map: Dict[int, str] = {}
        for item in j.get("mega_members", []):
            mid = int(item.get("mega_id", -1))
            mem = item.get("members", [])
            dirs = sorted(
                set(
                    str(m.get("direction"))
                    for m in mem
                    if isinstance(m, dict) and m.get("direction") is not None
                )
            )
            label = directions_to_label9(dirs)
            if mid >= 0 and label is not None:
                node_map[mid] = label
        out[obj] = node_map
    return out


def directions_to_label9(dirs: List[str]) -> str | None:
    s = set(dirs)
    if s == {"front"}:
        return "front"
    if s == {"front", "front-oblique"}:
        return "front/front-oblique"
    if s == {"front-oblique"}:
        return "front-oblique"
    if s == {"front-oblique", "side"}:
        return "front-oblique/side"
    if s == {"side"}:
        return "side"
    if s == {"side", "back-oblique"}:
        return "side/back-oblique"
    if s == {"back-oblique"}:
        return "back-oblique"
    if s == {"back-oblique", "back"}:
        return "back-oblique/back"
    if s == {"back"}:
        return "back"
    return None


def label5_to_label9(d: str) -> str | None:
    if d in DIRS_ORDER_5:
        return d
    return None


def format_count_ratio_lines(counts: Dict[str, int], order: List[str]) -> List[str]:
    total = int(sum(counts.values()))
    lines = []
    for k in order:
        v = int(counts.get(k, 0))
        pct = 100.0 * v / max(1, total)
        lines.append(f"  - {k}: {v} ({pct:.2f}%)")
    lines.append(f"  - total_nodes: {total}")
    return lines


def load_rep_node_distribution(rep_root: Path) -> Dict[str, Dict[str, int]]:
    reps_base = rep_root / "reps"
    out: Dict[str, Dict[str, int]] = {}
    if not reps_base.is_dir():
        return out

    for obj_dir in sorted(reps_base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        cnt = Counter()
        for d in DIRS_ORDER_5:
            p = obj_dir / d / "rep_k16.json"
            if not p.is_file():
                cnt[d] += 0
                continue
            try:
                j = read_json(p)
                cnt[d] += len(j.get("nodes", []))
            except Exception:
                cnt[d] += 0
        out[obj] = dict(cnt)
    return out


def load_objectgraph_node_distribution(mega_root: Path) -> Dict[str, Dict[str, int]]:
    base = mega_root / "megagraphs"
    out: Dict[str, Dict[str, int]] = {}
    if not base.is_dir():
        return out

    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        p = obj_dir / "megagraph_members.json"
        cnt = Counter()
        if p.is_file():
            try:
                j = read_json(p)
                for item in j.get("mega_members", []):
                    mem = item.get("members", [])
                    dirs = sorted(
                        set(
                            str(m.get("direction"))
                            for m in mem
                            if isinstance(m, dict) and m.get("direction") is not None
                        )
                    )
                    label = directions_to_label9(dirs)
                    if label is not None:
                        cnt[label] += 1
            except Exception:
                pass
        out[obj] = dict(cnt)
    return out


def evaluate_structure_classification(
    graphs_root: Path,
    alpha: float,
    topk: int,
    out_txt: Path,
) -> Dict[str, Any]:
    ot = import_ot()
    fgw2 = try_import_gromov_fn("fused_gromov_wasserstein2")
    if fgw2 is None:
        raise RuntimeError("fused_gromov_wasserstein2 not available")

    train_graphs, labels = collect_structure_train_templates(graphs_root)
    qpaths = list_jsons(graphs_root / "test")

    correct1 = 0
    correct5 = 0
    known = 0
    t0 = time.perf_counter()

    for qp in tqdm(qpaths, desc="structure cls", leave=False, dynamic_ncols=True):
        try:
            xq, _coordq, eq, true_obj, _true_dir = load_graph_json(qp)
        except Exception:
            continue
        cq = build_structure_cost(xq.shape[0], eq)
        pq = uniform_p(xq.shape[0])

        dists = []
        for _tp, xt, ct, pt, obj in train_graphs:
            try:
                d = dist_balanced(ot, fgw2, xt, ct, pt, xq, cq, pq, alpha=alpha)
            except Exception:
                d = float("inf")
            dists.append((float(d), obj))
        dists.sort(key=lambda x: x[0])

        score = defaultdict(lambda: float("inf"))
        for d, obj in dists:
            if d < score[obj]:
                score[obj] = d

        ranked = sorted(score.items(), key=lambda x: x[1])
        pred_objs = [o for o, _ in ranked[:topk]]
        if true_obj in labels:
            known += 1
            if pred_objs and pred_objs[0] == true_obj:
                correct1 += 1
            if true_obj in pred_objs[: min(5, len(pred_objs))]:
                correct5 += 1

    t1 = time.perf_counter()
    top1 = float(correct1 / max(1, known))
    top5 = float(correct5 / max(1, known))

    text = []
    text.append("Structure graph classification")
    text.append(f"graphs_root: {graphs_root}")
    text.append(f"alpha: {alpha}")
    text.append(f"top1: {top1:.6f}")
    text.append(f"top5: {top5:.6f}")
    text.append(f"known: {known}")
    text.append(f"time_sec: {t1 - t0:.6f}")
    write_text(out_txt, "\n".join(text) + "\n")

    return {"top1": top1, "top5": top5, "known": known, "time_sec": float(t1 - t0)}


def evaluate_rep_classification(
    rep_root: Path,
    graphs_root: Path,
    mode: str,
    alpha: float,
    partial_mass: float,
    topk: int,
    out_txt: Path,
) -> Dict[str, Any]:
    ot = import_ot()
    fgw2 = try_import_gromov_fn("fused_gromov_wasserstein2")
    pfgw2 = try_import_gromov_fn("partial_fused_gromov_wasserstein2")
    if mode == "balanced" and fgw2 is None:
        raise RuntimeError("fused_gromov_wasserstein2 not available")
    if mode == "partial" and pfgw2 is None:
        raise RuntimeError("partial_fused_gromov_wasserstein2 not available")

    rep_cache, labels = collect_rep_templates(rep_root)
    qpaths = list_jsons(graphs_root / "test")

    correct1 = 0
    correct5 = 0
    known = 0
    t0 = time.perf_counter()

    for qp in tqdm(qpaths, desc="representative cls", leave=False, dynamic_ncols=True):
        try:
            xq, _coordq, eq, true_obj, _true_dir = load_graph_json(qp)
        except Exception:
            continue
        cq = build_structure_cost(xq.shape[0], eq)
        pq = uniform_p(xq.shape[0])

        dists = []
        for obj in labels:
            best = float("inf")
            for _rp, xt, ct, pt, _direction in rep_cache.get(obj, []):
                try:
                    if mode == "balanced":
                        d = dist_balanced(ot, fgw2, xt, ct, pt, xq, cq, pq, alpha=alpha)
                    else:
                        d = dist_partial(ot, pfgw2, xt, ct, pt, xq, cq, pq, alpha=alpha, mass=partial_mass)
                except Exception:
                    d = float("inf")
                if d < best:
                    best = d
            dists.append((float(best), obj))
        dists.sort(key=lambda x: x[0])

        pred_objs = [o for _, o in dists[:topk]]
        if true_obj in labels:
            known += 1
            if pred_objs and pred_objs[0] == true_obj:
                correct1 += 1
            if true_obj in pred_objs[: min(5, len(pred_objs))]:
                correct5 += 1

    t1 = time.perf_counter()
    top1 = float(correct1 / max(1, known))
    top5 = float(correct5 / max(1, known))

    text = []
    text.append("Representative graph classification")
    text.append(f"rep_root: {rep_root}")
    text.append(f"graphs_root: {graphs_root}")
    text.append(f"mode: {mode}")
    text.append(f"alpha: {alpha}")
    text.append(f"partial_mass: {partial_mass}")
    text.append(f"top1: {top1:.6f}")
    text.append(f"top5: {top5:.6f}")
    text.append(f"known: {known}")
    text.append(f"time_sec: {t1 - t0:.6f}")
    write_text(out_txt, "\n".join(text) + "\n")

    return {"top1": top1, "top5": top5, "known": known, "time_sec": float(t1 - t0)}


def evaluate_object_classification_from_inference(
    inference_json: Path,
    out_txt: Path,
) -> Dict[str, Any]:
    j = read_json(inference_json)
    s = j.get("summary", {})
    t = j.get("timing", {})

    top1 = float(s.get("top1", 0.0))
    top5 = float(s.get("top5", 0.0))
    known = int(s.get("eval_known", 0))
    time_sec = t.get("num_queries", None)

    text = []
    text.append("Object graph classification (main run result)")
    text.append(f"inference_json: {inference_json}")
    text.append(f"top1: {top1:.6f}")
    text.append(f"top5: {top5:.6f}")
    text.append(f"known: {known}")
    if time_sec is not None:
        text.append(f"timing_info_num_queries: {time_sec}")
    write_text(out_txt, "\n".join(text) + "\n")

    return {"top1": top1, "top5": top5, "known": known}


def evaluate_object_classification_fgw(
    mega_root: Path,
    graphs_root: Path,
    alpha: float,
    topk: int,
    out_txt: Path,
) -> Dict[str, Any]:
    ot = import_ot()
    fgw2 = try_import_gromov_fn("fused_gromov_wasserstein2")
    if fgw2 is None:
        raise RuntimeError("fused_gromov_wasserstein2 not available")

    mega_cache = load_megagraphs(mega_root)
    labels = sorted(mega_cache.keys())
    qpaths = list_jsons(graphs_root / "test")

    correct1 = 0
    correct5 = 0
    known = 0
    t0 = time.perf_counter()

    for qp in tqdm(qpaths, desc="object cls FGW", leave=False, dynamic_ncols=True):
        try:
            xq, _coordq, eq, true_obj, _true_dir = load_graph_json(qp)
        except Exception:
            continue
        cq = build_structure_cost(xq.shape[0], eq)
        pq = uniform_p(xq.shape[0])

        dists = []
        for obj in labels:
            _gp, xt, ct, pt = mega_cache[obj]
            try:
                d = dist_balanced(ot, fgw2, xt, ct, pt, xq, cq, pq, alpha=alpha)
            except Exception:
                d = float("inf")
            dists.append((float(d), obj))
        dists.sort(key=lambda x: x[0])

        pred_objs = [o for _, o in dists[:topk]]
        if true_obj in labels:
            known += 1
            if pred_objs and pred_objs[0] == true_obj:
                correct1 += 1
            if true_obj in pred_objs[: min(5, len(pred_objs))]:
                correct5 += 1

    t1 = time.perf_counter()
    top1 = float(correct1 / max(1, known))
    top5 = float(correct5 / max(1, known))

    text = []
    text.append("Object graph classification with FGW")
    text.append(f"mega_root: {mega_root}")
    text.append(f"graphs_root: {graphs_root}")
    text.append(f"alpha: {alpha}")
    text.append(f"top1: {top1:.6f}")
    text.append(f"top5: {top5:.6f}")
    text.append(f"known: {known}")
    text.append(f"time_sec: {t1 - t0:.6f}")
    write_text(out_txt, "\n".join(text) + "\n")

    return {"top1": top1, "top5": top5, "known": known, "time_sec": float(t1 - t0)}


def evaluate_rep_direction_prediction(
    rep_root: Path,
    graphs_root: Path,
    mode: str,
    alpha: float,
    partial_mass: float,
    out_txt: Path,
    out_png: Path,
    legacy_out_root: Path | None = None,
    legacy_temp: float = 1.0,
    legacy_topk: int = 5,
) -> Dict[str, Any]:
    ot = import_ot()
    fgw2 = try_import_gromov_fn("fused_gromov_wasserstein2")
    pfgw2 = try_import_gromov_fn("partial_fused_gromov_wasserstein2")
    if mode == "balanced" and fgw2 is None:
        raise RuntimeError("fused_gromov_wasserstein2 not available")
    if mode == "partial" and pfgw2 is None:
        raise RuntimeError("partial_fused_gromov_wasserstein2 not available")

    rep_cache, _labels_unused = collect_rep_templates(rep_root)
    qpaths = list_jsons(graphs_root / "test")
    dir_labels = list(DIRS_ORDER_5)
    dir2idx = {d: i for i, d in enumerate(dir_labels)}

    total = 0
    correct = 0
    per_obj_total = Counter()
    per_obj_correct = Counter()
    per_obj_cm: Dict[str, np.ndarray] = {
        obj: np.zeros((len(dir_labels), len(dir_labels)), dtype=np.int64)
        for obj in sorted(rep_cache.keys())
    }
    cm_all = np.zeros((len(dir_labels), len(dir_labels)), dtype=np.int64)
    legacy_rows: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    for qp in tqdm(qpaths, desc=f"rep direction {mode}", leave=False, dynamic_ncols=True):
        try:
            xq, _coordq, eq, true_obj, true_dir = load_graph_json(qp)
        except Exception:
            continue
        if true_dir not in dir2idx:
            continue
        if true_obj not in rep_cache:
            continue

        cq = build_structure_cost(xq.shape[0], eq)
        pq = uniform_p(xq.shape[0])

        dists = np.full((len(dir_labels),), np.inf, dtype=np.float64)
        for i, dlabel in enumerate(dir_labels):
            for _rp, xt, ct, pt, direction in rep_cache.get(true_obj, []):
                if direction != dlabel:
                    continue
                try:
                    if mode == "balanced":
                        dist = dist_balanced(ot, fgw2, xt, ct, pt, xq, cq, pq, alpha=alpha)
                    else:
                        dist = dist_partial(ot, pfgw2, xt, ct, pt, xq, cq, pq, alpha=alpha, mass=partial_mass)
                except Exception:
                    dist = float("inf")
                dists[i] = min(dists[i], float(dist))

        if not np.any(np.isfinite(dists)):
            continue

        probs = softmax_from_negdist(dists, temp=legacy_temp)
        pred_idx = int(np.argmax(probs))
        pred_dir = dir_labels[pred_idx]

        ti = dir2idx[true_dir]
        pi = dir2idx[pred_dir]
        per_obj_cm.setdefault(
            true_obj,
            np.zeros((len(dir_labels), len(dir_labels)), dtype=np.int64),
        )[ti, pi] += 1
        cm_all[ti, pi] += 1
        total += 1
        per_obj_total[true_obj] += 1
        if ti == pi:
            correct += 1
            per_obj_correct[true_obj] += 1

        if legacy_out_root is not None:
            k = min(int(legacy_topk), len(dir_labels))
            top_idx = np.argsort(-probs)[:k].tolist()
            legacy_rows.append({
                "path": str(qp),
                "true_obj": true_obj,
                "true_dir": true_dir,
                "pred_dir": pred_dir,
                "top_dirs": [dir_labels[i] for i in top_idx],
                "top_probs": [float(probs[i]) for i in top_idx],
                "probs": [float(x) for x in probs],
            })

    t1 = time.perf_counter()
    micro_acc = float(correct / max(1, total))

    objs_for_report = sorted(set(list(rep_cache.keys()) + list(per_obj_total.keys())))
    per_object_summary: Dict[str, Dict[str, Any]] = {}
    accs = []
    for obj in objs_for_report:
        tot = int(per_obj_total.get(obj, 0))
        cor = int(per_obj_correct.get(obj, 0))
        obj_acc = float(cor / max(1, tot)) if tot > 0 else 0.0
        per_object_summary[obj] = {"accuracy": obj_acc, "correct": cor, "total": tot}
        accs.append(obj_acc)
    macro_acc = float(np.mean(accs)) if accs else 0.0

    plot_confusion_grid_by_object(
        per_obj_cm=per_obj_cm,
        labels=dir_labels,
        out_png=out_png,
        title="Representative graph direction prediction (object-known, 5-way per object)",
        ncols=4,
    )

    text = []
    text.append("Representative graph direction prediction (object-known)")
    text.append(f"rep_root: {rep_root}")
    text.append(f"graphs_root: {graphs_root}")
    text.append(f"mode: {mode}")
    text.append(f"alpha: {alpha}")
    if mode == "partial":
        text.append(f"partial_mass: {partial_mass}")
    text.append(f"micro_accuracy: {micro_acc:.6f}")
    text.append(f"macro_accuracy: {macro_acc:.6f}")
    text.append(f"total: {total}")
    text.append(f"labels: {dir_labels}")
    text.append(f"time_sec: {t1 - t0:.6f}")
    text.append("")
    text.append("[Per-object direction accuracy]")
    for obj in objs_for_report:
        s = per_object_summary[obj]
        text.append(
            f"- {obj}: accuracy={s['accuracy']:.6f} correct={int(s['correct'])} total={int(s['total'])}"
        )

    write_text(out_txt, "\n".join(text) + "\n")

    legacy_summary = None
    if legacy_out_root is not None:
        dir_out = legacy_out_root / "dir_only"
        ensure_dir(dir_out)
        write_text(dir_out / "classes.txt", "\n".join(dir_labels) + "\n")
        save_jsonl(dir_out / "preds.jsonl", legacy_rows)
        plot_confusion_legacy(cm_all, dir_labels, dir_out / "confusion_raw.png", "Confusion (direction-only, object-known)", normalize=False)
        plot_confusion_legacy(cm_all, dir_labels, dir_out / "confusion_norm.png", "Confusion (direction-only, object-known)", normalize=True)
        top5 = 1.0 if total > 0 else 0.0
        legacy_summary = {
            "mode": mode,
            "feature_mode": "dino",
            "alpha": float(alpha),
            "partial_mass": float(partial_mass),
            "temp": float(legacy_temp),
            "topk": int(legacy_topk),
            "dir_only": {
                "classes": len(dir_labels),
                "known": int(total),
                "top1": float(micro_acc),
                "top5": float(top5),
                "elapsed_sec": float(t1 - t0),
            },
        }
        write_json(legacy_out_root / "summary.json", legacy_summary)

    return {
        "accuracy": micro_acc,
        "micro_accuracy": micro_acc,
        "macro_accuracy": macro_acc,
        "total": total,
        "per_object": per_object_summary,
        "time_sec": float(t1 - t0),
        "mode": mode,
        "legacy_outputs": str(legacy_out_root) if legacy_out_root is not None else None,
        "legacy_summary": legacy_summary,
    }

def evaluate_object_direction_prediction(
    mega_root: Path,
    graphs_root: Path,
    mode: str,
    alpha: float,
    partial_mass: float,
    out_txt: Path,
    out_png: Path,
) -> Dict[str, Any]:
    ot = import_ot()
    fgw = try_import_gromov_fn("fused_gromov_wasserstein")
    pfgw = try_import_gromov_fn("partial_fused_gromov_wasserstein")
    fgw2 = try_import_gromov_fn("fused_gromov_wasserstein2")
    pfgw2 = try_import_gromov_fn("partial_fused_gromov_wasserstein2")
    if mode == "balanced" and (fgw is None or fgw2 is None):
        raise RuntimeError("balanced FGW functions not available")
    if mode == "partial" and (pfgw is None or pfgw2 is None):
        raise RuntimeError("partial FGW functions not available")

    mega_cache = load_megagraphs(mega_root)
    mega_label9 = load_megagraph_member_label9(mega_root)
    mega_node_dist = load_objectgraph_node_distribution(mega_root)

    labels = sorted(mega_cache.keys())
    qpaths = list_jsons(graphs_root / "test")

    dir2idx = {d: i for i, d in enumerate(DIRS_ORDER_9)}
    cm = np.zeros((len(DIRS_ORDER_9), len(DIRS_ORDER_9)), dtype=np.int64)
    total = 0
    correct = 0

    per_obj_total = Counter()
    per_obj_correct = Counter()

    t0 = time.perf_counter()
    for qp in tqdm(qpaths, desc="object direction", leave=False, dynamic_ncols=True):
        try:
            xq, _coordq, eq, true_obj, true_dir5 = load_graph_json(qp)
        except Exception:
            continue

        true_dir9 = label5_to_label9(true_dir5)
        if true_dir9 not in dir2idx:
            continue

        cq = build_structure_cost(xq.shape[0], eq)
        pq = uniform_p(xq.shape[0])

        best_obj = None
        best_dist = float("inf")

        for obj in labels:
            _gp, xt, ct, pt = mega_cache[obj]
            try:
                if mode == "balanced":
                    d = dist_balanced(ot, fgw2, xt, ct, pt, xq, cq, pq, alpha=alpha)
                else:
                    d = dist_partial(ot, pfgw2, xt, ct, pt, xq, cq, pq, alpha=alpha, mass=partial_mass)
            except Exception:
                d = float("inf")
            if d < best_dist:
                best_dist = d
                best_obj = obj

        if best_obj is None:
            continue

        _gp, xt, ct, pt = mega_cache[best_obj]
        try:
            if mode == "balanced":
                best_gamma = coupling_balanced(ot, fgw, xt, ct, pt, xq, cq, pq, alpha=alpha)
            else:
                best_gamma = coupling_partial(ot, pfgw, xt, ct, pt, xq, cq, pq, alpha=alpha, mass=partial_mass)
        except Exception:
            continue

        if best_gamma is None or best_gamma.size == 0:
            continue

        if best_gamma.shape[0] == xt.shape[0]:
            mega_mass = best_gamma.sum(axis=1)
        else:
            mega_mass = best_gamma.sum(axis=0)

        s = float(mega_mass.sum())
        if s > 0:
            mega_mass = mega_mass / s

        score9 = defaultdict(float)
        label_map = mega_label9.get(best_obj, {})
        for mid, mv in enumerate(mega_mass.tolist()):
            label9 = label_map.get(int(mid))
            if label9 is None:
                continue
            score9[label9] += float(mv)

        if not score9:
            continue

        pred_dir9 = max(score9.items(), key=lambda kv: kv[1])[0]
        if pred_dir9 not in dir2idx:
            continue

        ti = dir2idx[true_dir9]
        pi = dir2idx[pred_dir9]
        cm[ti, pi] += 1
        total += 1
        per_obj_total[true_obj] += 1
        if ti == pi:
            correct += 1
            per_obj_correct[true_obj] += 1

    t1 = time.perf_counter()
    acc = float(correct / max(1, total))
    plot_confusion(cm, DIRS_ORDER_9, out_png, title="Object graph direction prediction (9-way)")

    text = []
    text.append("Object graph direction prediction")
    text.append(f"mega_root: {mega_root}")
    text.append(f"graphs_root: {graphs_root}")
    text.append(f"mode: {mode}")
    text.append(f"alpha: {alpha}")
    text.append(f"partial_mass: {partial_mass}")
    text.append(f"accuracy: {acc:.6f}")
    text.append(f"total: {total}")
    text.append(f"labels: {DIRS_ORDER_9}")
    text.append(f"time_sec: {t1 - t0:.6f}")
    text.append("")

    text.append("[Per-object accuracy]")
    for obj in sorted(set(list(per_obj_total.keys()) + list(mega_node_dist.keys()))):
        tot = int(per_obj_total.get(obj, 0))
        cor = int(per_obj_correct.get(obj, 0))
        obj_acc = float(cor / max(1, tot)) if tot > 0 else 0.0
        text.append(f"- {obj}: accuracy={obj_acc:.6f} correct={cor} total={tot}")
    text.append("")

    text.append("[Per-object object-graph node distribution (9-way)]")
    for obj in sorted(mega_node_dist.keys()):
        text.append(f"- {obj}")
        text.extend(format_count_ratio_lines(mega_node_dist[obj], DIRS_ORDER_9))
        text.append("")

    write_text(out_txt, "\n".join(text) + "\n")
    return {"accuracy": acc, "total": total, "time_sec": float(t1 - t0)}


def run_tokencut_off_experiment(
    config_path: Path,
    out_root: Path,
    seed: int,
) -> Path:
    cfg = load_yaml(config_path)
    cfg.setdefault("graph", {})
    cfg["graph"]["fg_mode"] = "full"

    tmp_dir = Path(tempfile.mkdtemp(prefix="tokencut_off_cfg_"))
    tmp_cfg = tmp_dir / "tokencut_off.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    cmd = [
        "python",
        str((config_path.parent.parent.parent / "train.py").resolve()),
        "--config",
        str(tmp_cfg),
        "--seeds",
        str(seed),
    ]
    run_cmd(cmd)

    k = int(cfg.get("graph", {}).get("k", 16))
    run_dir = Path(cfg["out_root"]).resolve() / f"seed_{seed}_k_{k}"
    if not run_dir.is_dir():
        raise RuntimeError(f"TokenCut-off run dir not found: {run_dir}")

    dst_run = out_root / "tokencut_off_run"
    if dst_run.exists():
        shutil.rmtree(dst_run)
    shutil.copytree(run_dir, dst_run)
    shutil.rmtree(tmp_dir)
    return dst_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-tokencut-off", action="store_true")
    ap.add_argument("--skip-structure-classification", action="store_true")
    ap.add_argument("--rep-direction-mode", choices=["partial", "balanced"], default="balanced")
    ap.add_argument("--save-legacy-rep-direction-outputs", action="store_true")
    ap.add_argument("--legacy-rep-direction-temp", type=float, default=1.0)
    ap.add_argument("--legacy-rep-direction-topk", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    paths = infer_run_paths(run_dir)
    graphs_root = paths["graphs_root"]
    rep_root = paths["reps_root"]
    mega_root = paths["megagraph_root"]
    inference_json = paths["inference_json"]
    cfg = load_yaml(paths["config"])

    alpha = float(cfg.get("infer", {}).get("alpha", cfg.get("rep", {}).get("alpha", 0.3)))
    partial_mass = float(cfg.get("infer", {}).get("partial_mass", 0.7))
    topk = int(cfg.get("infer", {}).get("topk", 5))
    seed = int(cfg.get("graph", {}).get("seed", 0))

    results: Dict[str, Any] = {}

    total_steps = (4 if args.skip_structure_classification else 5) + (1 if inference_json.is_file() else 0) + (1 if args.run_tokencut_off else 0)
    with tqdm(total=total_steps, desc="ablation analysis", dynamic_ncols=True) as pbar:
        if args.skip_structure_classification:
            results["structure_classification"] = {"skipped": True}
        else:
            pbar.set_postfix(step="structure_classification")
            results["structure_classification"] = evaluate_structure_classification(
                graphs_root=graphs_root,
                alpha=alpha,
                topk=topk,
                out_txt=out_dir / "structure_classification.txt",
            )
            pbar.update(1)

        pbar.set_postfix(step="representative_classification")
        results["representative_classification"] = evaluate_rep_classification(
            rep_root=rep_root,
            graphs_root=graphs_root,
            mode="partial",
            alpha=alpha,
            partial_mass=partial_mass,
            topk=topk,
            out_txt=out_dir / "representative_classification.txt",
        )
        pbar.update(1)

        if inference_json.is_file():
            pbar.set_postfix(step="object_classification_main")
            results["object_classification_main"] = evaluate_object_classification_from_inference(
                inference_json=inference_json,
                out_txt=out_dir / "object_graph_classification_main.txt",
            )
            pbar.update(1)

        pbar.set_postfix(step="object_classification_fgw")
        results["object_classification_fgw"] = evaluate_object_classification_fgw(
            mega_root=mega_root,
            graphs_root=graphs_root,
            alpha=alpha,
            topk=topk,
            out_txt=out_dir / "object_graph_classification_fgw.txt",
        )
        pbar.update(1)

        pbar.set_postfix(step="representative_direction")
        results["representative_direction"] = evaluate_rep_direction_prediction(
            rep_root=rep_root,
            graphs_root=graphs_root,
            mode=args.rep_direction_mode,
            alpha=alpha,
            partial_mass=partial_mass,
            out_txt=out_dir / "direction_rep_results.txt",
            out_png=out_dir / "direction_rep_confusion.png",
            legacy_out_root=(out_dir / "direction_rep_legacy") if args.save_legacy_rep_direction_outputs else None,
            legacy_temp=args.legacy_rep_direction_temp,
            legacy_topk=args.legacy_rep_direction_topk,
        )
        pbar.update(1)

        pbar.set_postfix(step="object_direction")
        results["object_direction"] = evaluate_object_direction_prediction(
            mega_root=mega_root,
            graphs_root=graphs_root,
            mode="partial",
            alpha=alpha,
            partial_mass=partial_mass,
            out_txt=out_dir / "direction_object_results.txt",
            out_png=out_dir / "direction_object_confusion.png",
        )
        pbar.update(1)

        if args.run_tokencut_off:
            pbar.set_postfix(step="tokencut_off")
            tok_run_dir = run_tokencut_off_experiment(paths["config"], out_dir, seed)
            tok_paths = infer_run_paths(tok_run_dir)
            if tok_paths["inference_json"].is_file():
                results["tokencut_off_classification"] = evaluate_object_classification_from_inference(
                    inference_json=tok_paths["inference_json"],
                    out_txt=out_dir / "tokencut_off_classification.txt",
                )
            pbar.update(1)

    write_json(out_dir / "ablation_summary.json", results)
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
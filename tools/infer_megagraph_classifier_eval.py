#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MODE = "partial"
FEATURE_MODE = "dino"
MIN_NODES = 1
SAVE_MATCH = True


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def patch_scipy_for_pot():
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


@dataclass
class AnyGraph:
    path: str
    object_name: str
    direction: str
    feat: np.ndarray
    coord: np.ndarray
    edges: List[Tuple[int, int, float]]


def parse_obj_dir_from_path(p: Path) -> Tuple[str, str]:
    cand = p.parent.name
    if "_" in cand:
        obj = cand.split("_")[0]
        direction = "_".join(cand.split("_")[1:])
        return obj, direction
    return "unknown", "unknown"


def load_graph_json_allow_empty(path: Path, assume_pascal3d: bool = True) -> Tuple[Optional[AnyGraph], str]:
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"json_error: {e}"

    nodes = j.get("nodes", None)
    edges = j.get("edges", [])
    if nodes is None:
        return None, "missing_nodes_key"
    if len(nodes) == 0:
        return None, "empty_nodes"

    feat = np.stack([np.asarray(n["feat"], dtype=np.float64) for n in nodes], axis=0)
    coord = np.stack([np.array([float(n["x_norm"]), float(n["y_norm"])], dtype=np.float64) for n in nodes], axis=0)

    e_list = []
    for e in edges:
        try:
            u = int(e["u"])
            v = int(e["v"])
            d = float(e.get("dist", 1.0))
        except Exception:
            continue
        if u == v:
            continue
        if u > v:
            u, v = v, u
        e_list.append((u, v, d))

    obj, dname = parse_obj_dir_from_path(path) if assume_pascal3d else ("unknown", "unknown")
    return AnyGraph(str(path), obj, dname, feat, coord, e_list), "ok"


def build_structure_cost(n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    if not edges:
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


def build_node_features(g: AnyGraph) -> np.ndarray:
    return g.feat


def uniform_p(n: int) -> np.ndarray:
    p = np.ones((n,), dtype=np.float64)
    return p / max(p.sum(), 1e-12)


def list_jsons(root: Path) -> List[Path]:
    return sorted(root.rglob("*.json"))


def load_megagraphs(mega_root: Path) -> Dict[str, AnyGraph]:
    base = mega_root / "megagraphs"
    mp: Dict[str, AnyGraph] = {}
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        p = obj_dir / "megagraph.json"
        if not p.is_file():
            continue
        g, st = load_graph_json_allow_empty(p, assume_pascal3d=False)
        if g is None:
            continue
        g.object_name = obj
        mp[obj] = g
    if not mp:
        raise RuntimeError("No megagraphs found.")
    return mp


def load_megagraph_members(mega_root: Path) -> Dict[str, Any]:
    base = mega_root / "megagraphs"
    mp = {}
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        p = obj_dir / "megagraph_members.json"
        if p.is_file():
            try:
                mp[obj] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return mp


def build_dir_prop_maps(members_json: Any) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    if not isinstance(members_json, dict):
        return out
    mm_list = members_json.get("mega_members", [])
    if not isinstance(mm_list, list):
        return out
    for item in mm_list:
        if not isinstance(item, dict) or "mega_id" not in item:
            continue
        mid = int(item["mega_id"])
        mem = item.get("members", [])
        cnt: Dict[str, int] = {}
        if isinstance(mem, list):
            for m in mem:
                if not isinstance(m, dict):
                    continue
                dn = m.get("direction")
                if dn is None:
                    continue
                dn = str(dn).replace("_", "-")
                cnt[dn] = cnt.get(dn, 0) + 1
        tot = sum(cnt.values())
        if tot > 0:
            out[mid] = {k: cnt[k] / tot for k in cnt.keys()}
    return out


def dist_partial(ot, pfgw2, x1, c1, p, x2, c2, q, alpha: float, m: float) -> float:
    mm = ot.dist(x1, x2, metric="euclidean") ** 2
    d = call_with_supported_kwargs(
        pfgw2,
        M=mm,
        C1=c1,
        C2=c2,
        p=p,
        q=q,
        m=float(m),
        loss_fun="square_loss",
        alpha=float(alpha),
        log=False,
    )
    return float(d)


def coupling_partial(ot, pfgw, x1, c1, p, x2, c2, q, alpha: float, m: float) -> np.ndarray:
    mm = ot.dist(x1, x2, metric="euclidean") ** 2
    return np.asarray(
        call_with_supported_kwargs(
            pfgw,
            M=mm,
            C1=c1,
            C2=c2,
            p=p,
            q=q,
            m=float(m),
            loss_fun="square_loss",
            alpha=float(alpha),
            log=False,
        ),
        dtype=np.float64,
    )


def mass_on_mega(gamma: np.ndarray, n_mega: int) -> np.ndarray:
    if gamma.shape[0] == n_mega:
        mass = gamma.sum(axis=1)
    elif gamma.shape[1] == n_mega:
        mass = gamma.sum(axis=0)
    else:
        mass = gamma.sum(axis=1)
        if mass.shape[0] != n_mega:
            mass = np.resize(mass, (n_mega,))
    mass = np.asarray(mass, dtype=np.float64)
    s = float(mass.sum())
    if s > 0:
        mass = mass / s
    return mass


def plot_confusion(cm: np.ndarray, labels: List[str], out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = cm.astype(np.float64)
    row = cm.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    cmn = cm / row
    plt.figure(figsize=(9, 7))
    plt.imshow(cmn, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-graphs-root", required=True)
    ap.add_argument("--mega-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--partial-mass", type=float, default=0.7)
    ap.add_argument("--max-queries", type=int, default=0)
    args = ap.parse_args()

    ot = import_ot()
    pfgw2 = try_import_gromov_fn("partial_fused_gromov_wasserstein2")
    pfgw = try_import_gromov_fn("partial_fused_gromov_wasserstein")

    if pfgw2 is None:
        raise RuntimeError("partial_fused_gromov_wasserstein2 not available")

    mega_root = Path(args.mega_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mega = load_megagraphs(mega_root)
    mega_members = load_megagraph_members(mega_root) if SAVE_MATCH else {}
    labels = sorted(mega.keys())
    t1 = time.perf_counter()

    mega_cache = {}
    dir_prop_cache: Dict[str, Dict[int, Dict[str, float]]] = {}
    for obj in labels:
        g = mega[obj]
        xm = build_node_features(g)
        cm = build_structure_cost(g.feat.shape[0], g.edges)
        pm = uniform_p(xm.shape[0])
        mega_cache[obj] = (xm, cm, pm)
        if SAVE_MATCH:
            dir_prop_cache[obj] = build_dir_prop_maps(mega_members.get(obj, {}))

    qpaths = list_jsons(Path(args.query_graphs_root))
    if args.max_queries and len(qpaths) > args.max_queries:
        qpaths = qpaths[:args.max_queries]

    cmat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    label2idx = {lb: i for i, lb in enumerate(labels)}
    correct1 = 0
    correct5 = 0
    known = 0
    skipped = 0

    predictions: List[Dict[str, Any]] = []
    skipped_items: List[Dict[str, Any]] = []

    for qi, qp in enumerate(qpaths, 1):
        t_q0 = time.perf_counter()
        qg, st = load_graph_json_allow_empty(qp, assume_pascal3d=True)

        if qg is None:
            skipped += 1
            _, dg = parse_obj_dir_from_path(qp)
            skipped_items.append(
                {
                    "path": str(qp),
                    "status": st,
                    "reason": "load_failed_or_empty",
                    "direction_guess": dg,
                }
            )
            continue

        if qg.feat.shape[0] < MIN_NODES:
            skipped += 1
            skipped_items.append(
                {
                    "path": str(qp),
                    "status": "too_few_nodes",
                    "reason": f"nodes={qg.feat.shape[0]}",
                    "direction_guess": qg.direction,
                }
            )
            continue

        xq = build_node_features(qg)
        cq = build_structure_cost(qg.feat.shape[0], qg.edges)
        pqm = uniform_p(xq.shape[0])

        dists: List[Tuple[float, str]] = []
        for obj in labels:
            xm, cm_obj, pm = mega_cache[obj]
            try:
                d = dist_partial(ot, pfgw2, xm, cm_obj, pm, xq, cq, pqm, alpha=args.alpha, m=args.partial_mass)
            except Exception:
                d = float("inf")
            dists.append((float(d), obj))
        dists.sort(key=lambda x: x[0])

        topk_eff = max(1, min(args.topk, len(dists)))
        top_objs = [o for _, o in dists[:topk_eff]]
        top_ds = [float(d) for d, _ in dists[:topk_eff]]
        pred1 = top_objs[0]
        dist1 = top_ds[0]

        gt_dist = None
        if qg.object_name in mega_cache:
            for d, o in dists:
                if o == qg.object_name:
                    gt_dist = float(d) if np.isfinite(d) else None
                    break

        wrong = [(d, o) for (d, o) in dists if o != qg.object_name and np.isfinite(d)]
        neg1_obj = wrong[0][1] if len(wrong) >= 1 else None
        neg1_dist = float(wrong[0][0]) if len(wrong) >= 1 else None
        neg2_obj = wrong[1][1] if len(wrong) >= 2 else None
        neg2_dist = float(wrong[1][0]) if len(wrong) >= 2 else None

        match_info = None
        if SAVE_MATCH:
            xm, cm_obj, pm = mega_cache[pred1]
            gamma = None
            try:
                if pfgw is not None:
                    gamma = coupling_partial(ot, pfgw, xm, cm_obj, pm, xq, cq, pqm, alpha=args.alpha, m=args.partial_mass)
            except Exception:
                gamma = None

            if gamma is not None and gamma.size > 0:
                mass = mass_on_mega(gamma, n_mega=xm.shape[0])
                prop_map = dir_prop_cache.get(pred1, {})

                dir_mass: Dict[str, float] = {}
                for mid, mv in enumerate(mass):
                    if mv <= 0:
                        continue
                    prop = prop_map.get(mid)
                    if not prop:
                        continue
                    for dn, frac in prop.items():
                        dir_mass[dn] = dir_mass.get(dn, 0.0) + float(mv) * float(frac)

                top_ids = np.argsort(-mass).tolist()
                top_nodes = [
                    {
                        "mega_id": int(i),
                        "mass": float(mass[i]),
                        "dir_prop": prop_map.get(int(i)),
                    }
                    for i in top_ids
                    if float(mass[i]) > 0
                ]
                match_info = {"top_nodes": top_nodes, "dir_mass": dir_mass}

        t_q1 = time.perf_counter()
        elapsed = float(t_q1 - t_q0)

        is_correct = 1 if (pred1 == qg.object_name) else 0

        predictions.append(
            {
                "path": str(qp),
                "true_object": qg.object_name,
                "true_direction": qg.direction,
                "pred_top1": pred1,
                "pred_topk_objects": top_objs,
                "pred_topk_dists": top_ds,
                "pred_top1_dist": float(dist1) if np.isfinite(dist1) else None,
                "is_correct_top1": int(is_correct),
                "gt_object_dist": gt_dist,
                "neg1_object": neg1_obj,
                "neg1_dist": neg1_dist,
                "neg2_object": neg2_obj,
                "neg2_dist": neg2_dist,
                "num_nodes": int(qg.feat.shape[0]),
                "elapsed_query_sec": elapsed,
                "match_info": match_info,
            }
        )

        if qg.object_name in label2idx:
            known += 1
            ti = label2idx[qg.object_name]
            pi = label2idx.get(pred1, None)
            if pi is not None:
                cmat[ti, pi] += 1
            if pred1 == qg.object_name:
                correct1 += 1
            if qg.object_name in top_objs[:min(5, len(top_objs))]:
                correct5 += 1

        if qi % 50 == 0 or qi == len(qpaths):
            print(f"[{MODE}] {qi}/{len(qpaths)} skipped={skipped}")

    top1 = float(correct1 / max(1, known))
    top5 = float(correct5 / max(1, known))

    confusion_path = out_root / "confusion.png"
    if known > 0:
        plot_confusion(cmat, labels, confusion_path, title=f"Confusion ({MODE})")

    inference_result = {
        "settings": {
            "mode": MODE,
            "feature_mode": FEATURE_MODE,
            "min_nodes": int(MIN_NODES),
            "save_match": bool(SAVE_MATCH),
            "alpha": float(args.alpha),
            "topk": int(args.topk),
            "partial_mass": float(args.partial_mass),
            "max_queries": int(args.max_queries),
            "query_graphs_root": str(args.query_graphs_root),
            "mega_root": str(args.mega_root),
            "out_root": str(args.out_root),
        },
        "timing": {
            "offline_load_sec": float(t1 - t0),
            "num_queries": int(len(qpaths)),
        },
        "summary": {
            "mode": MODE,
            "num_queries_total": int(len(qpaths)),
            "num_skipped": int(skipped),
            "eval_known": int(known),
            "top1": top1,
            "top5": top5,
            "alpha": float(args.alpha),
            "partial_mass": float(args.partial_mass),
            "confusion_png": str(confusion_path),
        },
        "predictions": predictions,
        "skipped": skipped_items,
    }

    save_json(out_root / "inference_result.json", inference_result)
    print(f"[DONE] top1={top1:.4f} top5={top5:.4f} out={out_root}")

if __name__ == "__main__":
    main()
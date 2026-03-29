#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

K_TARGET = 16
FULL_MEDOID_THRESHOLD = 120


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


def import_pot():
    try:
        import ot
        from ot.gromov import fused_gromov_wasserstein2, fused_gromov_wasserstein
        return ot, fused_gromov_wasserstein2, fused_gromov_wasserstein
    except Exception:
        patch_scipy_for_pot()
        import ot
        from ot.gromov import fused_gromov_wasserstein2, fused_gromov_wasserstein
        return ot, fused_gromov_wasserstein2, fused_gromov_wasserstein


@dataclass
class GraphData:
    feat_dino: np.ndarray
    coord: np.ndarray
    cluster: np.ndarray
    edge_index: np.ndarray
    edge_dist: np.ndarray
    direction: str
    object_name: str
    path: str
    node_weight: Optional[np.ndarray] = None


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_single_graph_json(path: Path, direction: str, object_name: str) -> GraphData:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    nodes = obj.get("nodes", [])
    edges = obj.get("edges", [])
    if len(nodes) == 0:
        raise ValueError("empty nodes")

    feat_dino = np.stack([np.asarray(n["feat"], dtype=np.float32) for n in nodes], axis=0)
    coord = np.stack([np.array([float(n["x_norm"]), float(n["y_norm"])], dtype=np.float32) for n in nodes], axis=0)
    cluster = np.array([int(n.get("cluster", 0)) for n in nodes], dtype=np.int64)

    weights = []
    has_w = False
    for n in nodes:
        if "size" in n:
            has_w = True
            weights.append(float(n["size"]))
        else:
            weights.append(1.0)
    node_weight = np.array(weights, dtype=np.float32) if has_w else None

    ei_u, ei_v, ew = [], [], []
    for e in edges:
        u = int(e["u"])
        v = int(e["v"])
        d = float(e.get("dist", 1.0))
        ei_u += [u, v]
        ei_v += [v, u]
        ew += [d, d]

    if len(ei_u) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_dist = np.zeros((0,), dtype=np.float32)
    else:
        edge_index = np.vstack([
            np.array(ei_u, dtype=np.int64),
            np.array(ei_v, dtype=np.int64),
        ])
        edge_dist = np.asarray(ew, dtype=np.float32)

    return GraphData(
        feat_dino=feat_dino,
        coord=coord,
        cluster=cluster,
        edge_index=edge_index,
        edge_dist=edge_dist,
        direction=direction,
        object_name=object_name,
        path=str(path),
        node_weight=node_weight,
    )


def parse_object_direction(dir_name: str) -> Tuple[str, str]:
    if "_" not in dir_name:
        return dir_name, "unknown"
    obj, d = dir_name.split("_", 1)
    return obj, d


def discover_objects_and_dirs(split_root: Path) -> Dict[str, List[Tuple[str, Path]]]:
    mp: Dict[str, List[Tuple[str, Path]]] = {}
    for p in sorted(split_root.iterdir()):
        if not p.is_dir():
            continue
        obj, d = parse_object_direction(p.name)
        mp.setdefault(obj, []).append((d, p))
    return mp


def load_direction_graphs_from_dir(dir_path: Path, direction: str, object_name: str) -> List[GraphData]:
    files = sorted(dir_path.rglob("*.json"))
    gs: List[GraphData] = []
    for f in files:
        try:
            gs.append(load_single_graph_json(f, direction=direction, object_name=object_name))
        except Exception:
            continue
    return gs


def build_node_features(g: GraphData) -> np.ndarray:
    return g.feat_dino.astype(np.float32)


def build_structure_cost(g: GraphData) -> np.ndarray:
    n = g.feat_dino.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if g.edge_dist.size == 0 or g.edge_index.shape[1] == 0:
        return np.zeros((n, n), dtype=np.float32)

    c = np.zeros((n, n), dtype=np.float32)
    max_d = float(g.edge_dist.max()) if g.edge_dist.size else 1.0
    max_d = max(max_d, 1e-6)
    c[:] = max_d

    ei = g.edge_index
    dists = g.edge_dist
    for k in range(ei.shape[1]):
        u = int(ei[0, k])
        v = int(ei[1, k])
        w = float(dists[k])
        if w < c[u, v]:
            c[u, v] = w
    np.fill_diagonal(c, 0.0)
    c = 0.5 * (c + c.T)
    return c.astype(np.float32)


def node_distribution(g: GraphData) -> np.ndarray:
    n = g.feat_dino.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if g.node_weight is None:
        p = np.ones((n,), dtype=np.float64)
    else:
        p = g.node_weight.astype(np.float64).copy()
    s = p.sum()
    if s <= 0:
        p[:] = 1.0
        s = p.sum()
    return p / s


def fgw_coupling(ot, fgw, xs, cs, ps, xt, ct, pt, alpha: float) -> np.ndarray:
    m = ot.dist(xs, xt, metric="euclidean") ** 2
    gamma = fgw(m, cs, ct, ps, pt, loss_fun="square_loss", alpha=alpha, log=False)
    return np.asarray(gamma, dtype=np.float64)


def fgw_distance(ot, fgw2, x1, c1, p, x2, c2, q, alpha: float) -> float:
    m = ot.dist(x1, x2, metric="euclidean") ** 2
    d = fgw2(m, c1, c2, p, q, loss_fun="square_loss", alpha=alpha)
    return float(d)


def choose_pivots_even(n: int, m: int) -> List[int]:
    if m >= n:
        return list(range(n))
    idx = np.linspace(0, n - 1, m, dtype=int)
    return sorted(list(set(idx.tolist())))


def choose_anchor_fgw_all(ot, fgw2, graphs_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]], alpha: float) -> int:
    n = len(graphs_cache)
    if n == 1:
        return 0
    sumd = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        xi, ci, pi, _, _ = graphs_cache[i]
        for j in range(i + 1, n):
            xj, cj, pj, _, _ = graphs_cache[j]
            try:
                d = fgw_distance(ot, fgw2, xi, ci, pi, xj, cj, pj, alpha=alpha)
            except Exception:
                d = float("inf")
            sumd[i] += d
            sumd[j] += d
    return int(np.argmin(sumd))


def compute_stability_metrics(
    ot,
    fgw,
    anchor_x: np.ndarray,
    anchor_c: np.ndarray,
    anchor_p: np.ndarray,
    graphs_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    alpha: float,
) -> Dict[str, np.ndarray]:
    n0, d = anchor_x.shape
    sum_conc = np.zeros((n0,), dtype=np.float64)
    sum_ent = np.zeros((n0,), dtype=np.float64)

    used = 0
    mean_feat = np.zeros((n0, d), dtype=np.float64)
    m2_feat = np.zeros((n0, d), dtype=np.float64)

    for x, c, p, _coord, _path in graphs_cache:
        try:
            gamma = fgw_coupling(ot, fgw, anchor_x, anchor_c, anchor_p, x, c, p, alpha=alpha)
        except Exception:
            continue

        row = np.maximum(gamma.sum(axis=1), 1e-12)
        w = gamma / row[:, None]

        conc = w.max(axis=1)
        ww = np.clip(w, 1e-12, 1.0)
        ent = -(ww * np.log(ww)).sum(axis=1)

        sum_conc += conc
        sum_ent += ent

        mu_feat = w @ x
        used += 1

        df = mu_feat - mean_feat
        mean_feat += df / used
        m2_feat += df * (mu_feat - mean_feat)

    if used == 0:
        raise RuntimeError("No successful couplings for stability.")

    conc_mean = sum_conc / used
    ent_mean = sum_ent / used
    var_feat = (m2_feat / max(used - 1, 1)).sum(axis=1)

    return {
        "conc_mean": conc_mean,
        "ent_mean": ent_mean,
        "var_feat": var_feat,
        "used": np.array([used], dtype=np.int64),
    }


def rank_sum_select_topk(metrics: Dict[str, np.ndarray], k: int) -> np.ndarray:
    conc = metrics["conc_mean"]
    ent = metrics["ent_mean"]
    vf = metrics["var_feat"]
    n = conc.shape[0]
    k = min(k, n)

    r1 = np.argsort(np.argsort(-conc))
    r2 = np.argsort(np.argsort(ent))
    r3 = np.argsort(np.argsort(vf))

    rs = r1 + r2 + r3
    keep_idx = np.argsort(rs)[:k]
    return np.sort(keep_idx)


def build_rep_from_fixed_support(
    ot,
    fgw,
    rep_x: np.ndarray,
    rep_c: np.ndarray,
    rep_p: np.ndarray,
    graphs_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n0, d = rep_x.shape
    x_acc = np.zeros((n0, d), dtype=np.float64)
    coord_acc = np.zeros((n0, 2), dtype=np.float64)
    c_acc = np.zeros((n0, n0), dtype=np.float64)
    used = 0

    for x, c, p, coord, _path in graphs_cache:
        try:
            gamma = fgw_coupling(ot, fgw, rep_x, rep_c, rep_p, x, c, p, alpha=alpha)
        except Exception:
            continue

        row = np.maximum(gamma.sum(axis=1), 1e-12)
        x_map = (gamma @ x) / row[:, None]
        coord_map = (gamma @ coord) / row[:, None]
        denom = np.maximum(row[:, None] * row[None, :], 1e-12)
        c_map = (gamma @ c @ gamma.T) / denom

        x_acc += x_map
        coord_acc += coord_map
        c_acc += c_map
        used += 1

    if used == 0:
        raise RuntimeError("No successful couplings for representative.")

    x_rep = x_acc / used
    coord_rep = coord_acc / used
    c_rep = c_acc / used
    np.fill_diagonal(c_rep, 0.0)
    c_rep = 0.5 * (c_rep + c_rep.T)
    c_rep = np.clip(c_rep, 0.0, None)

    p_rep = rep_p.copy()
    p_rep = p_rep / max(p_rep.sum(), 1e-12)
    return x_rep, c_rep, coord_rep, p_rep, used


def save_rep_json(out_path: Path, x_rep: np.ndarray, c_rep: np.ndarray, coord_rep: np.ndarray, dino_dim: int, meta: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = x_rep.shape[0]
    feat_dino = x_rep[:, :dino_dim].astype(np.float32)
    nodes = []
    for i in range(n):
        nodes.append(
            {
                "id": int(i),
                "feat": feat_dino[i].astype(float).tolist(),
                "x_norm": float(coord_rep[i, 0]),
                "y_norm": float(coord_rep[i, 1]),
                "cluster": 0,
            }
        )

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append({"u": int(u), "v": int(v), "dist": float(c_rep[u, v])})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges, "meta": meta}, f, ensure_ascii=False, indent=2)


def build_one_direction(
    ot,
    fgw2,
    fgw,
    train_graphs: List[GraphData],
    alpha: float,
):
    dino_dim = int(train_graphs[0].feat_dino.shape[1])

    train_cache = []
    for g in train_graphs:
        x = build_node_features(g).astype(np.float64)
        c = build_structure_cost(g).astype(np.float64)
        p = node_distribution(g).astype(np.float64)
        coord = g.coord.astype(np.float64)
        train_cache.append((x, c, p, coord, g.path))

    anchor_idx = choose_anchor_fgw_all(ot, fgw2, train_cache, alpha=alpha)
    anchor_x, anchor_c, anchor_p, anchor_coord, anchor_path = train_cache[anchor_idx]

    metrics = compute_stability_metrics(
        ot,
        fgw,
        anchor_x,
        anchor_c,
        anchor_p,
        train_cache,
        alpha=alpha,
    )

    keep_idx = rank_sum_select_topk(metrics, k=K_TARGET)

    rep0_x = anchor_x[keep_idx]
    rep0_c = anchor_c[np.ix_(keep_idx, keep_idx)]
    rep0_coord = anchor_coord[keep_idx]
    rep0_p = np.ones((rep0_x.shape[0],), dtype=np.float64)
    rep0_p /= rep0_p.sum()

    xr, cr, coordr, pr, used_train = build_rep_from_fixed_support(
        ot, fgw, rep0_x, rep0_c, rep0_p, train_cache, alpha=alpha
    )

    pack = dict(
        X=xr,
        C=cr,
        coord=coordr,
        p=pr,
        dino_dim=dino_dim,
        meta=dict(
            method="anchor_one_shot",
            alpha=float(alpha),
            k_target=int(K_TARGET),
            anchor_path=str(anchor_path),
            anchor_index=int(anchor_idx),
            num_train_graphs=int(len(train_graphs)),
            used_train=int(used_train),
            anchor_num_nodes=int(anchor_x.shape[0]),
            selected_support_indices=keep_idx.astype(int).tolist(),
        ),
    )
    return pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fgw-alpha", type=float, default=0.5)
    args = ap.parse_args()

    ot, fgw2, fgw = import_pot()

    graphs_root = Path(args.graphs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root = graphs_root / "train"
    train_map = discover_objects_and_dirs(train_root)
    if not train_map:
        raise RuntimeError(f"no object_direction dirs under {train_root}")

    reps_root = out_dir / "reps"
    summary_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    t_all0 = time.time()

    for obj in sorted(train_map.keys()):
        for d, train_dir in train_map[obj]:
            train_graphs = load_direction_graphs_from_dir(train_dir, direction=d, object_name=obj)
            if len(train_graphs) == 0:
                errors.append({
                    "object": obj,
                    "direction": d,
                    "reason": "train_graphs_empty",
                    "detail": str(train_dir),
                })
                continue

            t0 = time.time()
            try:
                rep = build_one_direction(
                    ot,
                    fgw2,
                    fgw,
                    train_graphs=train_graphs,
                    alpha=args.fgw_alpha,
                )
            except Exception as e:
                errors.append({
                    "object": obj,
                    "direction": d,
                    "reason": "build_rep_failed",
                    "detail": repr(e),
                })
                continue

            elapsed = time.time() - t0
            rep_json = reps_root / obj / d / "rep_k16.json"

            save_rep_json(
                rep_json,
                rep["X"],
                rep["C"],
                rep["coord"],
                dino_dim=rep["dino_dim"],
                meta={**rep["meta"], "object": obj, "direction": d},
            )

            summary_rows.append(
                {
                    "object": obj,
                    "direction": d,
                    "rep_json_path": str(rep_json),
                    "num_train_graphs": int(len(train_graphs)),
                    "rep_num_nodes": int(rep["X"].shape[0]),
                    "elapsed_sec": float(elapsed),
                    "alpha": float(args.fgw_alpha),
                    "anchor_path": str(rep["meta"]["anchor_path"]),
                    "anchor_index": int(rep["meta"]["anchor_index"]),
                    "used_train": int(rep["meta"]["used_train"]),
                }
            )
            print(f"[OK] {obj}:{d} time={elapsed:.1f}s")

    summary = {
        "graphs_root": str(graphs_root),
        "out_dir": str(out_dir),
        "train_root": str(train_root),
        "alpha": float(args.fgw_alpha),
        "k_target": int(K_TARGET),
        "num_reps_built": int(len(summary_rows)),
        "num_errors": int(len(errors)),
        "elapsed_total_sec": float(time.time() - t_all0),
        "reps": summary_rows,
        "errors": errors,
    }

    save_json(out_dir / "rep_build_summary.json", summary)
    print(f"[DONE] reps={summary['num_reps_built']} errors={summary['num_errors']}")
    print(f"[SUMMARY] {out_dir / 'rep_build_summary.json'}")


if __name__ == "__main__":
    main()
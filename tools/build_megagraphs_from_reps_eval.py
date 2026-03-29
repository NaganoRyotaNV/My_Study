#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

DIRS_ORDER = ["front", "front-oblique", "side", "back-oblique", "back"]
SCORE_MODE = "bilateral"
CANDIDATE_SIDE = "both"
MAX_SPAN = 1


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
        from ot.gromov import fused_gromov_wasserstein
        return ot, fused_gromov_wasserstein
    except Exception:
        patch_scipy_for_pot()
        import ot
        from ot.gromov import fused_gromov_wasserstein
        return ot, fused_gromov_wasserstein


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class RepGraph:
    obj: str
    direction: str
    path: str
    feat: np.ndarray
    coord: np.ndarray
    cmat: np.ndarray


def load_rep(path: Path, obj: str, direction: str) -> RepGraph:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    nodes = j.get("nodes", [])
    edges = j.get("edges", [])
    if not nodes:
        raise ValueError(f"empty nodes: {path}")

    feat = np.stack([np.asarray(n["feat"], np.float32) for n in nodes], axis=0)
    coord = np.stack([np.array([float(n["x_norm"]), float(n["y_norm"])], np.float32) for n in nodes], axis=0)

    n = feat.shape[0]
    cmat = np.zeros((n, n), dtype=np.float64)
    if edges:
        maxd = max(float(e.get("dist", 1.0)) for e in edges)
        maxd = max(maxd, 1e-6)
        cmat[:] = maxd
        np.fill_diagonal(cmat, 0.0)
        for e in edges:
            u = int(e["u"])
            v = int(e["v"])
            d = float(e.get("dist", 1.0))
            if d < cmat[u, v]:
                cmat[u, v] = d
                cmat[v, u] = d
        cmat = 0.5 * (cmat + cmat.T)
        np.fill_diagonal(cmat, 0.0)

    return RepGraph(
        obj=obj,
        direction=direction,
        path=str(path),
        feat=feat.astype(np.float64),
        coord=coord.astype(np.float64),
        cmat=cmat,
    )


def discover_reps(rep_root: Path) -> Dict[str, Dict[str, Path]]:
    base = rep_root / "reps"
    if not base.is_dir():
        raise RuntimeError(f"rep_root must contain reps/: {base}")

    mp: Dict[str, Dict[str, Path]] = {}
    for obj_dir in sorted(base.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        for ddir in sorted(obj_dir.iterdir()):
            if not ddir.is_dir():
                continue
            d = ddir.name
            p = ddir / "rep_k16.json"
            if p.is_file():
                mp.setdefault(obj, {})[d] = p
    return mp


def uniform_p(n: int) -> np.ndarray:
    p = np.ones((n,), dtype=np.float64)
    return p / max(p.sum(), 1e-12)


def fgw_gamma(ot, fgw, xa, ca, pa, xb, cb, pb, alpha: float) -> np.ndarray:
    m = ot.dist(xa, xb, metric="euclidean") ** 2
    g = fgw(m, ca, cb, pa, pb, loss_fun="square_loss", alpha=float(alpha), log=False)
    return np.asarray(g, dtype=np.float64)


def score_from_gamma(gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = np.maximum(gamma.sum(axis=1), 1e-12)
    col = np.maximum(gamma.sum(axis=0), 1e-12)
    w_row = gamma / row[:, None]
    w_col = gamma / col[None, :]
    score = np.sqrt(np.clip(w_row * w_col, 0.0, None))
    sa = score.max(axis=1)
    sb = score.max(axis=0)
    return score, sa, sb


def chain_pairs(dirs_order: List[str]) -> List[Tuple[str, str]]:
    return [(dirs_order[i], dirs_order[i + 1]) for i in range(len(dirs_order) - 1)]


def compute_global_threshold(
    ot,
    fgw,
    reps_index: Dict[str, Dict[str, Path]],
    alpha: float,
    top_frac: float,
) -> Tuple[float, int, int]:
    pairs = chain_pairs(DIRS_ORDER)
    all_s: List[float] = []
    used_objects = 0

    for obj, dirmap in reps_index.items():
        if any(d not in dirmap for d in DIRS_ORDER):
            continue
        used_objects += 1
        reps = {d: load_rep(dirmap[d], obj=obj, direction=d) for d in DIRS_ORDER}

        for da, db in pairs:
            a = reps[da]
            b = reps[db]
            pa = uniform_p(a.feat.shape[0])
            pb = uniform_p(b.feat.shape[0])

            try:
                gamma = fgw_gamma(ot, fgw, a.feat, a.cmat, pa, b.feat, b.cmat, pb, alpha=alpha)
            except Exception:
                continue

            _, sa, sb = score_from_gamma(gamma)
            all_s.extend(sa.tolist())
            all_s.extend(sb.tolist())

    if not all_s:
        raise RuntimeError("No values collected for threshold.")

    q = 1.0 - float(top_frac)
    t = float(np.quantile(np.asarray(all_s, np.float64), q))
    return t, len(all_s), used_objects


class UnionFindSpan:
    def __init__(self, n: int, pos: List[int], max_span: int):
        self.p = list(range(n))
        self.r = [0] * n
        self.min_pos = pos[:]
        self.max_pos = pos[:]
        self.max_span = int(max_span)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def can_union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        new_min = min(self.min_pos[ra], self.min_pos[rb])
        new_max = max(self.max_pos[ra], self.max_pos[rb])
        return (new_max - new_min) <= self.max_span

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True

        new_min = min(self.min_pos[ra], self.min_pos[rb])
        new_max = max(self.max_pos[ra], self.max_pos[rb])
        if (new_max - new_min) > self.max_span:
            return False

        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
            self.min_pos[rb] = new_min
            self.max_pos[rb] = new_max
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
            self.min_pos[ra] = new_min
            self.max_pos[ra] = new_max
        else:
            self.p[rb] = ra
            self.r[ra] += 1
            self.min_pos[ra] = new_min
            self.max_pos[ra] = new_max
        return True


def build_megagraph_for_object(
    ot,
    fgw,
    obj: str,
    dirmap: Dict[str, Path],
    alpha: float,
    threshold: float,
) -> Tuple[Dict, Dict, Dict]:
    pairs = chain_pairs(DIRS_ORDER)
    reps = {d: load_rep(dirmap[d], obj=obj, direction=d) for d in DIRS_ORDER}

    offsets: Dict[str, int] = {}
    pos_of_global: List[int] = []
    total = 0
    for di, d in enumerate(DIRS_ORDER):
        offsets[d] = total
        k = reps[d].feat.shape[0]
        pos_of_global += [di] * k
        total += k

    uf = UnionFindSpan(total, pos=pos_of_global, max_span=MAX_SPAN)
    all_candidates: List[Dict] = []
    pair_stats_map: Dict[str, Dict[str, int]] = {}

    for da, db in pairs:
        pair_name = f"{da}-{db}"
        pair_stats_map[pair_name] = {
            "candidates": 0,
            "merged": 0,
            "blocked_by_span": 0,
            "blocked_by_used_node": 0,
            "blocked_by_same_component": 0,
        }

        a = reps[da]
        b = reps[db]
        pa = uniform_p(a.feat.shape[0])
        pb = uniform_p(b.feat.shape[0])
        gamma = fgw_gamma(ot, fgw, a.feat, a.cmat, pa, b.feat, b.cmat, pb, alpha=alpha)
        score, sa, sb = score_from_gamma(gamma)

        cand_pairs: Dict[Tuple[int, int], float] = {}

        for i in range(score.shape[0]):
            if sa[i] < threshold:
                continue
            j = int(np.argmax(score[i]))
            cand_pairs[(i, j)] = max(cand_pairs.get((i, j), -1.0), float(score[i, j]))

        for j in range(score.shape[1]):
            if sb[j] < threshold:
                continue
            i = int(np.argmax(score[:, j]))
            cand_pairs[(i, j)] = max(cand_pairs.get((i, j), -1.0), float(score[i, j]))

        for (i, j), sc in cand_pairs.items():
            gi = offsets[da] + i
            gj = offsets[db] + j
            all_candidates.append(
                {
                    "pair": pair_name,
                    "da": da,
                    "db": db,
                    "i": int(i),
                    "j": int(j),
                    "gi": int(gi),
                    "gj": int(gj),
                    "score": float(sc),
                    "left_strength": float(sa[i]),
                    "right_strength": float(sb[j]),
                }
            )
            pair_stats_map[pair_name]["candidates"] += 1

    all_candidates.sort(key=lambda x: x["score"], reverse=True)

    used_global_nodes = set()
    merge_debug: List[Dict] = []

    for item in all_candidates:
        pair_name = item["pair"]
        gi = item["gi"]
        gj = item["gj"]

        if gi in used_global_nodes or gj in used_global_nodes:
            pair_stats_map[pair_name]["blocked_by_used_node"] += 1
            continue

        if uf.find(gi) == uf.find(gj):
            pair_stats_map[pair_name]["blocked_by_same_component"] += 1
            continue

        if not uf.can_union(gi, gj):
            pair_stats_map[pair_name]["blocked_by_span"] += 1
            continue

        ok = uf.union(gi, gj)
        if not ok:
            pair_stats_map[pair_name]["blocked_by_span"] += 1
            continue

        used_global_nodes.add(gi)
        used_global_nodes.add(gj)
        pair_stats_map[pair_name]["merged"] += 1

        merge_debug.append(
            {
                "pair": pair_name,
                "da": item["da"],
                "db": item["db"],
                "i": int(item["i"]),
                "j": int(item["j"]),
                "gi": int(item["gi"]),
                "gj": int(item["gj"]),
                "score": float(item["score"]),
                "left_strength": float(item["left_strength"]),
                "right_strength": float(item["right_strength"]),
                "union_ok": True,
            }
        )

    per_pair_stats = []
    for da, db in pairs:
        pair_name = f"{da}-{db}"
        st = pair_stats_map[pair_name]
        per_pair_stats.append(
            {
                "pair": pair_name,
                "candidates": int(st["candidates"]),
                "merged": int(st["merged"]),
                "blocked_by_used_node": int(st["blocked_by_used_node"]),
                "blocked_by_span": int(st["blocked_by_span"]),
                "blocked_by_same_component": int(st["blocked_by_same_component"]),
                "threshold": float(threshold),
            }
        )

    comp: Dict[int, List[Tuple[str, int]]] = {}
    for d in DIRS_ORDER:
        k = reps[d].feat.shape[0]
        for i in range(k):
            g = offsets[d] + i
            r = uf.find(g)
            comp.setdefault(r, []).append((d, i))

    roots = sorted(comp.keys())
    root2mid = {r: idx for idx, r in enumerate(roots)}
    m = len(roots)
    dim = reps[DIRS_ORDER[0]].feat.shape[1]
    mega_feat = np.zeros((m, dim), dtype=np.float64)
    mega_coord = np.zeros((m, 2), dtype=np.float64)

    members_out = []
    for r in roots:
        mid = root2mid[r]
        mem = comp[r]
        feats = []
        coords = []
        mem_list = []
        for d, i in mem:
            feats.append(reps[d].feat[i])
            coords.append(reps[d].coord[i])
            mem_list.append({"direction": d, "local_node": int(i)})
        mega_feat[mid] = np.mean(np.stack(feats, axis=0), axis=0)
        mega_coord[mid] = np.mean(np.stack(coords, axis=0), axis=0)
        members_out.append({"mega_id": int(mid), "members": mem_list})

    edge_sum: Dict[Tuple[int, int], float] = {}
    edge_cnt: Dict[Tuple[int, int], int] = {}

    for d in DIRS_ORDER:
        cmat = reps[d].cmat
        k = cmat.shape[0]
        for u in range(k):
            for v in range(u + 1, k):
                mu = root2mid[uf.find(offsets[d] + u)]
                mv = root2mid[uf.find(offsets[d] + v)]
                if mu == mv:
                    continue
                a, b = (mu, mv) if mu < mv else (mv, mu)
                w = float(cmat[u, v])
                edge_sum[(a, b)] = edge_sum.get((a, b), 0.0) + w
                edge_cnt[(a, b)] = edge_cnt.get((a, b), 0) + 1

    edges = [
        {"u": int(a), "v": int(b), "dist": float(edge_sum[(a, b)] / edge_cnt[(a, b)])}
        for (a, b) in edge_sum.keys()
    ]
    edges.sort(key=lambda e: (e["u"], e["v"]))

    mega_json = {
        "nodes": [
            {
                "id": int(i),
                "feat": mega_feat[i].astype(np.float32).tolist(),
                "x_norm": float(mega_coord[i, 0]),
                "y_norm": float(mega_coord[i, 1]),
            }
            for i in range(m)
        ],
        "edges": edges,
        "meta": {
            "object": obj,
            "dirs_order": DIRS_ORDER,
            "adjacency": "CHAIN",
            "alpha": float(alpha),
            "threshold": float(threshold),
            "score_mode": SCORE_MODE,
            "candidate_side": CANDIDATE_SIDE,
            "max_span": int(MAX_SPAN),
            "pairs": [{"a": a, "b": b} for a, b in pairs],
            "merge_policy": "global_greedy_no_node_reuse_across_all_adjacent_pairs",
        },
    }

    members_json = {
        "object": obj,
        "dirs_order": DIRS_ORDER,
        "threshold": float(threshold),
        "score_mode": SCORE_MODE,
        "candidate_side": CANDIDATE_SIDE,
        "max_span": int(MAX_SPAN),
        "merge_policy": "global_greedy_no_node_reuse_across_all_adjacent_pairs",
        "pair_stats": per_pair_stats,
        "merge_debug": merge_debug,
        "mega_members": members_out,
    }

    summary = {
        "object": obj,
        "mega_nodes": int(m),
        "total_original_nodes": int(sum(reps[d].feat.shape[0] for d in DIRS_ORDER)),
        "total_merged_links": int(len(merge_debug)),
        "pair_stats": per_pair_stats,
    }

    return mega_json, members_json, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    ot, fgw = import_pot()

    rep_root = Path(args.rep_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    reps_index = discover_reps(rep_root)
    if not reps_index:
        raise RuntimeError(f"No reps found under {rep_root}/reps")

    t_thr0 = time.perf_counter()
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_mode = "manual"
        collected_values = None
        used_objects_for_threshold = None
    else:
        threshold, collected_values, used_objects_for_threshold = compute_global_threshold(
            ot=ot,
            fgw=fgw,
            reps_index=reps_index,
            alpha=float(args.alpha),
            top_frac=float(args.top_frac),
        )
        threshold_mode = "global"
    threshold_elapsed = time.perf_counter() - t_thr0

    summary_rows = []
    errors = []
    t_all0 = time.perf_counter()

    for obj in sorted(reps_index.keys()):
        dirmap = reps_index[obj]
        if any(d not in dirmap for d in DIRS_ORDER):
            errors.append(
                {
                    "object": obj,
                    "reason": "missing_direction_for_chain",
                    "detail": f"need={DIRS_ORDER} have={sorted(list(dirmap.keys()))}",
                }
            )
            continue

        t0 = time.perf_counter()
        try:
            mega, members, summ = build_megagraph_for_object(
                ot=ot,
                fgw=fgw,
                obj=obj,
                dirmap=dirmap,
                alpha=float(args.alpha),
                threshold=float(threshold),
            )
        except Exception as e:
            errors.append({"object": obj, "reason": "build_failed", "detail": repr(e)})
            continue

        elapsed = time.perf_counter() - t0
        summary_rows.append(
            {
                "object": obj,
                "mega_nodes": summ["mega_nodes"],
                "total_original_nodes": summ["total_original_nodes"],
                "total_merged_links": summ["total_merged_links"],
                "elapsed_sec": float(elapsed),
                "alpha": float(args.alpha),
                "threshold": float(threshold),
            }
        )

        obj_out = out_root / "megagraphs" / obj
        save_json(obj_out / "megagraph.json", mega)
        save_json(obj_out / "megagraph_members.json", members)
        print(f"[SAVED] {obj} mega_nodes={summ['mega_nodes']} merged_links={summ['total_merged_links']} time={elapsed:.2f}s")

    summary = {
        "rep_root": str(rep_root),
        "out_root": str(out_root),
        "dirs_order": DIRS_ORDER,
        "alpha": float(args.alpha),
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
        "top_frac": float(args.top_frac) if args.threshold is None else None,
        "collected_values": int(collected_values) if collected_values is not None else None,
        "used_objects_for_threshold": int(used_objects_for_threshold) if used_objects_for_threshold is not None else None,
        "threshold_elapsed_sec": float(threshold_elapsed),
        "num_objects_built": int(len(summary_rows)),
        "num_errors": int(len(errors)),
        "elapsed_total_sec": float(time.perf_counter() - t_all0),
        "objects": summary_rows,
        "errors": errors,
    }

    save_json(out_root / "megagraph_build_summary.json", summary)
    print(f"[DONE] objects={summary['num_objects_built']} errors={summary['num_errors']}")
    print(f"[SUMMARY] {out_root / 'megagraph_build_summary.json'}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import yaml
from matplotlib.patches import Circle, Wedge

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


DIRS_ORDER = ["front", "front-oblique", "side", "back-oblique", "back"]
DIR_COLOR = {
    "front": (1.0, 0.9, 0.0),
    "front-oblique": (0.2, 0.9, 0.2),
    "side": (0.2, 0.6, 1.0),
    "back-oblique": (1.0, 0.6, 0.0),
    "back": (0.9, 0.2, 0.2),
    "unknown": (0.7, 0.7, 0.7),
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    dst.write_bytes(src.read_bytes())


@dataclass
class GraphData:
    x: np.ndarray
    y: np.ndarray
    feat: Optional[np.ndarray]
    edges: List[Tuple[int, int, float]]
    meta: Dict[str, Any]


def load_graph_json(path: Path) -> GraphData:
    j = read_json(path)
    nodes = j.get("nodes", [])
    edges = j.get("edges", [])
    meta = j.get("meta", {}) if isinstance(j.get("meta", {}), dict) else {}

    if len(nodes) == 0:
        return GraphData(
            x=np.zeros((0,), np.float32),
            y=np.zeros((0,), np.float32),
            feat=None,
            edges=[],
            meta=meta,
        )

    x = np.array([float(n.get("x_norm", 0.0)) for n in nodes], dtype=np.float32)
    y = np.array([float(n.get("y_norm", 0.0)) for n in nodes], dtype=np.float32)

    feat = None
    if "feat" in nodes[0]:
        try:
            feat = np.stack([np.asarray(n["feat"], np.float32) for n in nodes], axis=0)
        except Exception:
            feat = None

    e_list: List[Tuple[int, int, float]] = []
    for e in edges:
        try:
            u = int(e["u"])
            v = int(e["v"])
            d = float(e.get("dist", 1.0))
            e_list.append((u, v, d))
        except Exception:
            continue

    return GraphData(x=x, y=y, feat=feat, edges=e_list, meta=meta)


def parse_config_effective(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config_effective.yaml"
    if not cfg_path.is_file():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def infer_k_from_run(run_dir: Path, cfg: Dict[str, Any]) -> int:
    if isinstance(cfg.get("graph"), dict) and "k" in cfg["graph"]:
        return int(cfg["graph"]["k"])

    graphs_dir = run_dir / "graphs"
    ks = []

    for p in graphs_dir.glob("k*"):
        m = re.fullmatch(r"k(\d+)", p.name)
        if m:
            ks.append(int(m.group(1)))

    for p in graphs_dir.rglob("*__k*.json"):
        m = re.search(r"__k(\d+)$", p.stem)
        if m:
            ks.append(int(m.group(1)))

    if not ks:
        raise RuntimeError(f"Could not infer k from: {graphs_dir}")
    return sorted(set(ks))[0]


def try_find_stageA_triptych_png(graph_json: Path) -> Optional[Path]:
    base = graph_json.stem
    base2 = re.sub(r"__k\d+$", "", base)
    parent = graph_json.parent
    candidates: List[Path] = []
    for d in [parent, parent / "viz", parent / "vis", parent.parent / "viz", parent.parent / "vis"]:
        if d.is_dir():
            for p in d.glob("*.png"):
                name = p.stem
                if base in name or base2 in name:
                    candidates.append(p)
    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def image_path_from_graph_json(graph_json: Path) -> Optional[Path]:
    try:
        j = read_json(graph_json)
        p = j.get("image_path")
        if isinstance(p, str) and p.strip():
            pp = Path(p)
            if pp.is_file():
                return pp
    except Exception:
        pass
    return None


def draw_graph_matplotlib(
    g: GraphData,
    out_png: Path,
    title: str = "",
    show_edges: bool = True,
    invert_y: bool = True,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
) -> None:
    ensure_dir(out_png.parent)
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    if show_edges and len(g.edges) > 0:
        for u, v, _d in g.edges:
            if u < 0 or v < 0 or u >= g.x.size or v >= g.x.size:
                continue
            ax.plot([g.x[u], g.x[v]], [g.y[u], g.y[v]], linewidth=0.8, alpha=0.35, color="black")

    ax.scatter(g.x, g.y, s=90)
    for i in range(g.x.size):
        ax.text(g.x[i], g.y[i], str(i), fontsize=10, ha="center", va="center")

    ax.set_xlabel("x_norm")
    ax.set_ylabel("y_norm")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if invert_y:
        ax.invert_yaxis()
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def draw_graph_overlay_on_image(
    img_bgr: np.ndarray,
    g: GraphData,
    out_png: Path,
    node_radius: int = 6,
    edge_thickness: int = 2,
) -> None:
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]

    for u, v, _d in g.edges:
        if u < 0 or v < 0 or u >= g.x.size or v >= g.x.size:
            continue
        x1 = int(round(float(g.x[u]) * (w - 1)))
        y1 = int(round(float(g.y[u]) * (h - 1)))
        x2 = int(round(float(g.x[v]) * (w - 1)))
        y2 = int(round(float(g.y[v]) * (h - 1)))
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), edge_thickness, lineType=cv2.LINE_AA)

    for i in range(g.x.size):
        cx = int(round(float(g.x[i]) * (w - 1)))
        cy = int(round(float(g.y[i]) * (h - 1)))
        cv2.circle(canvas, (cx, cy), node_radius + 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), node_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    ensure_dir(out_png.parent)
    cv2.imwrite(str(out_png), canvas)


def load_dir_mix_per_node(mega_root: Path, obj: str) -> Dict[int, Dict[str, float]]:
    p = mega_root / "megagraphs" / obj / "megagraph_members.json"
    if not p.is_file():
        return {}
    j = read_json(p)
    out: Dict[int, Dict[str, float]] = {}
    for item in j.get("mega_members", []):
        mid = int(item.get("mega_id", -1))
        mem = item.get("members", [])
        cnt = Counter(
            [m.get("direction") for m in mem if isinstance(m, dict) and m.get("direction") is not None]
        )
        tot = sum(cnt.values())
        if mid >= 0 and tot > 0:
            out[mid] = {d: cnt[d] / tot for d in cnt.keys()}
    return out


def edge_colors_by_component_clique(g: GraphData) -> List[Tuple[Tuple[int, int, float], str]]:
    n = int(g.x.size)
    if n == 0 or len(g.edges) == 0:
        return []

    adj = [set() for _ in range(n)]
    for u, v, _d in g.edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u].add(v)
            adj[v].add(u)

    comp_id = [-1] * n
    comps: List[List[int]] = []
    cid = 0
    for s in range(n):
        if comp_id[s] != -1:
            continue
        stack = [s]
        comp_id[s] = cid
        nodes: List[int] = []
        while stack:
            a = stack.pop()
            nodes.append(a)
            for b in adj[a]:
                if comp_id[b] == -1:
                    comp_id[b] = cid
                    stack.append(b)
        comps.append(nodes)
        cid += 1

    is_clique = [True] * len(comps)
    for i, nodes in enumerate(comps):
        m = len(nodes)
        if m <= 2:
            is_clique[i] = True
            continue
        node_set = set(nodes)
        ecount = 0
        for a in nodes:
            ecount += sum(1 for b in adj[a] if b in node_set)
        ecount //= 2
        is_clique[i] = ecount == (m * (m - 1)) // 2

    out: List[Tuple[Tuple[int, int, float], str]] = []
    for e in g.edges:
        u, v, d = e
        if not (0 <= u < n and 0 <= v < n):
            continue
        c = comp_id[u]
        color = "black" if (c >= 0 and is_clique[c]) else "yellow"
        out.append(((u, v, d), color))
    return out


def draw_megagraph_matplotlib(
    g: GraphData,
    out_png: Path,
    title: str,
    node_mix: Dict[int, Dict[str, float]],
    show_edges: bool = True,
    invert_y: bool = True,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
) -> None:
    ensure_dir(out_png.parent)
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    if show_edges and len(g.edges) > 0:
        colored_edges = edge_colors_by_component_clique(g)
        for (u, v, _d), col in colored_edges:
            if u < 0 or v < 0 or u >= g.x.size or v >= g.x.size:
                continue
            ax.plot(
                [g.x[u], g.x[v]],
                [g.y[u], g.y[v]],
                linewidth=1.0,
                alpha=0.85 if col == "yellow" else 0.45,
                color=col,
            )

    n = int(g.x.size)
    r = 0.018 if n <= 40 else 0.012

    for i in range(n):
        x = float(g.x[i])
        y = float(g.y[i])
        mix = node_mix.get(i, None)

        if not mix:
            ax.add_patch(Circle((x, y), r, facecolor=DIR_COLOR["unknown"], edgecolor="black", linewidth=1.0))
        else:
            items: List[Tuple[str, float]] = []
            for d in DIRS_ORDER:
                if d in mix:
                    items.append((d, float(mix[d])))
            for d in mix.keys():
                if d not in DIRS_ORDER:
                    items.append((d, float(mix[d])))

            s = sum(w for _, w in items)
            if s <= 0:
                ax.add_patch(Circle((x, y), r, facecolor=DIR_COLOR["unknown"], edgecolor="black", linewidth=1.0))
            else:
                items = [(d, w / s) for d, w in items]
                start = 0.0
                for d, frac in items:
                    end = start + 360.0 * frac
                    color = DIR_COLOR.get(d, DIR_COLOR["unknown"])
                    ax.add_patch(Wedge((x, y), r, start, end, facecolor=color, edgecolor="black", linewidth=0.6))
                    start = end
                ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor="black", linewidth=1.0))

        ax.text(x, y, str(i), fontsize=9, ha="center", va="center")

    import matplotlib.lines as mlines
    leg_items = []
    for d in DIRS_ORDER:
        leg_items.append(mlines.Line2D([], [], color=DIR_COLOR[d], marker="o", linestyle="None", markersize=8, label=d))
    leg_items.append(mlines.Line2D([], [], color="black", linewidth=2, label="edge (clique component)"))
    leg_items.append(mlines.Line2D([], [], color="yellow", linewidth=2, label="edge (non-clique component)"))
    ax.legend(handles=leg_items, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlabel("x_norm")
    ax.set_ylabel("y_norm")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if invert_y:
        ax.invert_yaxis()
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


GRAPH_KEYS = [
    "anchor_graph_json",
    "anchor_graph_path",
    "medoid_graph_json",
    "medoid_graph_path",
    "anchor_graph",
    "medoid_graph",
    "anchor_path",
    "medoid_path",
]


def get_anchor_graph_path_from_rep_meta(meta: Dict[str, Any]) -> Optional[Path]:
    for k in GRAPH_KEYS:
        v = meta.get(k, None)
        if isinstance(v, str) and v.strip():
            p = Path(v)
            if p.suffix.lower() == ".json" and p.is_file():
                return p
    return None


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


def uniform_p(n: int) -> np.ndarray:
    p = np.ones((n,), dtype=np.float64)
    return p / max(p.sum(), 1e-12)


def graph_json_to_cost_matrices(graph_json: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    j = read_json(graph_json)
    nodes = j.get("nodes", [])
    edges = j.get("edges", [])
    feat = np.stack([np.asarray(n["feat"], np.float64) for n in nodes], axis=0) if nodes else np.zeros((0, 1), np.float64)
    coord = np.stack(
        [np.array([float(n["x_norm"]), float(n["y_norm"])], np.float64) for n in nodes],
        axis=0,
    ) if nodes else np.zeros((0, 2), np.float64)

    n = feat.shape[0]
    if n == 0:
        return feat, coord, np.zeros((0, 0), np.float64)

    if not edges:
        return feat, coord, np.zeros((n, n), np.float64)

    max_d = max(float(e.get("dist", 1.0)) for e in edges)
    max_d = max(max_d, 1e-6)
    c = np.full((n, n), max_d, np.float64)
    np.fill_diagonal(c, 0.0)
    for e in edges:
        u = int(e["u"])
        v = int(e["v"])
        d = float(e.get("dist", 1.0))
        if 0 <= u < n and 0 <= v < n and d < c[u, v]:
            c[u, v] = d
            c[v, u] = d
    c = 0.5 * (c + c.T)
    np.fill_diagonal(c, 0.0)
    return feat, coord, c


def build_node_features(feat: np.ndarray, coord: np.ndarray) -> np.ndarray:
    return feat


def load_dir_prop(mega_root: Path, obj: str) -> Dict[int, Dict[str, float]]:
    p = mega_root / "megagraphs" / obj / "megagraph_members.json"
    if not p.is_file():
        return {}
    j = read_json(p)
    out: Dict[int, Dict[str, float]] = {}
    for item in j.get("mega_members", []):
        mid = int(item.get("mega_id", -1))
        mem = item.get("members", [])
        cnt = Counter([m.get("direction") for m in mem if isinstance(m, dict) and m.get("direction") is not None])
        tot = sum(cnt.values())
        if mid >= 0 and tot > 0:
            out[mid] = {k: cnt[k] / tot for k in cnt.keys()}
    return out


def plot_gamma(gamma: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(gamma, interpolation="nearest")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel("query nodes")
    plt.ylabel("mega nodes")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


def pfgw_distance_and_gamma(
    mega_graph_json: Path,
    query_graph_json: Path,
    alpha: float,
    mass: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    ot = import_ot()
    pfgw = try_import_gromov_fn("partial_fused_gromov_wasserstein")
    if pfgw is None:
        raise RuntimeError("ot.gromov.partial_fused_gromov_wasserstein is not available")

    mf, mc, mC = graph_json_to_cost_matrices(mega_graph_json)
    qf, qc, qC = graph_json_to_cost_matrices(query_graph_json)

    xm = build_node_features(mf, mc)
    xq = build_node_features(qf, qc)
    pm = uniform_p(xm.shape[0])
    pq = uniform_p(xq.shape[0])

    m = ot.dist(xm, xq, metric="euclidean") ** 2
    gamma = call_with_supported_kwargs(
        pfgw,
        M=m,
        C1=mC,
        C2=qC,
        p=pm,
        q=pq,
        m=float(mass),
        loss_fun="square_loss",
        alpha=float(alpha),
        log=False,
    )
    gamma = np.asarray(gamma, dtype=np.float64)

    row = gamma.sum(axis=1)
    col = gamma.sum(axis=0)
    if row.sum() > 0:
        row = row / row.sum()
    if col.sum() > 0:
        col = col / col.sum()

    dist_proxy = float((m * gamma).sum())
    return dist_proxy, gamma, row, col


def export_structure_subset(
    graphs_root: Path,
    out_dir: Path,
    split: str,
    max_images: int,
    k: int,
) -> None:
    if max_images <= 0:
        return

    src_root = graphs_root / split
    if not src_root.is_dir():
        return

    jsons = sorted(src_root.rglob(f"*__k{k}.json"))
    if len(jsons) == 0:
        jsons = sorted(src_root.rglob("*.json"))

    jsons = jsons[:max_images]

    for gj in tqdm(jsons, desc=f"structure/{split}"):
        rel = gj.relative_to(src_root)
        stem = gj.stem
        out_item = out_dir / "structure" / split / rel.parent / stem
        ensure_dir(out_item)

        g = load_graph_json(gj)

        img_path = image_path_from_graph_json(gj)
        if img_path is not None and img_path.is_file():
            safe_copy(img_path, out_item / f"1_input{img_path.suffix}")
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        else:
            img_bgr = None

        trip = try_find_stageA_triptych_png(gj)
        if trip is not None and trip.is_file():
            safe_copy(trip, out_item / "2_stageA_viz.png")

        if img_bgr is not None:
            draw_graph_overlay_on_image(img_bgr, g, out_item / "3_graph_overlay.png")

        title = f"{split}:{rel.parent.as_posix()}  {stem}"
        draw_graph_matplotlib(g, out_item / "4_graph_axes.png", title=title, show_edges=True)

        write_json(out_item / "meta.json", {"graph_json": str(gj), "image": str(img_path) if img_path else None})


def export_representatives_and_anchors(
    run_dir: Path,
    out_dir: Path,
    k: int,
) -> None:
    reps_root = run_dir / "reps" / "reps"
    if not reps_root.is_dir():
        return

    rep_jsons = sorted(reps_root.rglob(f"rep_*k{k}.json"))
    if len(rep_jsons) == 0:
        rep_jsons = sorted(reps_root.rglob("rep_*.json"))

    for repj in tqdm(rep_jsons, desc="representatives"):
        try:
            rep = load_graph_json(repj)
            meta = rep.meta if isinstance(rep.meta, dict) else {}
            rel = repj.relative_to(reps_root)
            obj = rel.parts[-3] if len(rel.parts) >= 3 else "unknown_obj"
            direction = rel.parts[-2] if len(rel.parts) >= 2 else "unknown_dir"

            rep_out = out_dir / "representative" / obj / direction
            ensure_dir(rep_out)

            title = f"{obj}:{direction} rep(K={k})"
            draw_graph_matplotlib(rep, rep_out / "rep_graph_axes.png", title=title, show_edges=True)

            anchor_gjson = get_anchor_graph_path_from_rep_meta(meta)
            if anchor_gjson is not None and anchor_gjson.is_file():
                anch_out = rep_out / "anchor"
                ensure_dir(anch_out)

                safe_copy(anchor_gjson, anch_out / anchor_gjson.name)

                anchor_img = image_path_from_graph_json(anchor_gjson)
                if anchor_img is not None and anchor_img.is_file():
                    safe_copy(anchor_img, anch_out / f"anchor_image{anchor_img.suffix}")
                    img_bgr = cv2.imread(str(anchor_img), cv2.IMREAD_COLOR)
                else:
                    img_bgr = None

                trip = try_find_stageA_triptych_png(anchor_gjson)
                if trip is not None and trip.is_file():
                    safe_copy(trip, anch_out / "anchor_stageA_viz.png")

                anch_g = load_graph_json(anchor_gjson)
                if img_bgr is not None:
                    draw_graph_overlay_on_image(img_bgr, anch_g, anch_out / "anchor_overlay.png")
                draw_graph_matplotlib(
                    anch_g,
                    anch_out / "anchor_graph_axes.png",
                    title=f"Anchor: {obj}:{direction}",
                    show_edges=True,
                )

                write_json(
                    anch_out / "anchor_meta.json",
                    {
                        "rep_json": str(repj),
                        "anchor_graph_json": str(anchor_gjson),
                        "anchor_image": str(anchor_img) if anchor_img else None,
                    },
                )
        except Exception:
            continue


def count_rep_nodes_for_object(run_dir: Path, obj: str) -> int:
    reps_root = run_dir / "reps" / "reps" / obj
    if not reps_root.is_dir():
        return 0
    total = 0
    for p in sorted(reps_root.glob("*/rep_k16.json")):
        try:
            j = read_json(p)
            total += len(j.get("nodes", []))
        except Exception:
            continue
    return total


def export_object_graphs(run_dir: Path, out_dir: Path) -> None:
    mega_root = run_dir / "megagraphs"
    mg_root = mega_root / "megagraphs"
    if not mg_root.is_dir():
        return

    objs = sorted([d.name for d in mg_root.iterdir() if d.is_dir()])

    for obj in tqdm(objs, desc="object graphs"):
        gj = mg_root / obj / "megagraph.json"
        if not gj.is_file():
            continue

        g = load_graph_json(gj)
        node_mix = load_dir_mix_per_node(mega_root, obj)

        before_nodes = count_rep_nodes_for_object(run_dir, obj)
        after_nodes = int(g.x.size)

        outp = out_dir / "object" / obj
        ensure_dir(outp)

        draw_megagraph_matplotlib(
            g,
            outp / "object_graph_axes.png",
            title=f"ObjectGraph: {obj} ({before_nodes} -> {after_nodes} nodes)",
            node_mix=node_mix,
            show_edges=True,
        )
        safe_copy(gj, outp / "megagraph.json")
        write_json(
            outp / "object_meta.json",
            {
                "object": obj,
                "rep_total_nodes_before_merge": before_nodes,
                "megagraph_nodes_after_merge": after_nodes,
            },
        )


def export_pfgw_top5_from_inference(
    run_dir: Path,
    out_dir: Path,
    n_queries: int,
    alpha: float,
    mass: float,
) -> None:
    if n_queries <= 0:
        return

    infer_json = run_dir / "inference" / "inference_result.json"
    mega_root = run_dir / "megagraphs"
    mg_root = mega_root / "megagraphs"

    if not infer_json.is_file() or not mg_root.is_dir():
        return

    inf = read_json(infer_json)
    preds = inf.get("predictions", [])
    if not isinstance(preds, list) or len(preds) == 0:
        return

    preds = preds[:n_queries]
    csv_rows: List[Dict[str, Any]] = []

    for item in tqdm(preds, desc="pfgw top5"):
        qpath = item.get("path")
        if not isinstance(qpath, str) or not Path(qpath).is_file():
            continue

        qj = Path(qpath)
        qstem = qj.stem
        obj_dir = qj.parent.name
        outp = out_dir / "pfgw_top5" / obj_dir / qstem
        ensure_dir(outp)

        qimg = image_path_from_graph_json(qj)
        if qimg is not None and qimg.is_file():
            safe_copy(qimg, outp / f"query_image{qimg.suffix}")

        top_objs = item.get("pred_topk_objects", [])
        top_dists = item.get("pred_topk_dists", [])
        gt_obj = item.get("true_object")

        top5_list = []
        for i in range(min(5, len(top_objs))):
            top5_list.append(
                {
                    "rank": i + 1,
                    "object": top_objs[i],
                    "dist": float(top_dists[i]) if i < len(top_dists) else None,
                }
            )

        best_gamma_info = None
        if len(top_objs) > 0:
            best_obj = top_objs[0]
            best_gj = mg_root / best_obj / "megagraph.json"
            if best_gj.is_file():
                try:
                    dist_proxy, gamma, row, col = pfgw_distance_and_gamma(
                        best_gj,
                        qj,
                        alpha=alpha,
                        mass=mass,
                    )
                    dir_prop = load_dir_prop(mega_root, best_obj)
                    dir_attention: Dict[str, float] = {}
                    for i, w in enumerate(row):
                        if w <= 0:
                            continue
                        prop = dir_prop.get(i)
                        if not prop:
                            continue
                        for d, frac in prop.items():
                            dir_attention[d] = dir_attention.get(d, 0.0) + float(w) * float(frac)

                    top_m = np.argsort(-row)[: min(30, row.size)].tolist()
                    top_q = np.argsort(-col)[: min(30, col.size)].tolist()
                    sub = gamma[np.ix_(top_m, top_q)]
                    plot_gamma(sub, outp / "gamma_heatmap_top1.png", title=f"gamma top nodes ({best_obj})")

                    best_gamma_info = {
                        "object": best_obj,
                        "dist_proxy": float(dist_proxy),
                        "dir_attention": {k: float(v) for k, v in sorted(dir_attention.items())},
                    }
                except Exception:
                    best_gamma_info = None

        write_json(
            outp / "top5.json",
            {
                "query_graph": str(qj),
                "query_image": str(qimg) if qimg else None,
                "true_object": gt_obj,
                "top5": top5_list,
                "top1_gamma_info": best_gamma_info,
                "raw_prediction_entry": item,
            },
        )

        row_out: Dict[str, Any] = {
            "query_graph": str(qj),
            "true_object": gt_obj,
        }
        for i in range(5):
            if i < len(top_objs):
                row_out[f"top{i+1}_object"] = top_objs[i]
                row_out[f"top{i+1}_dist"] = float(top_dists[i]) if i < len(top_dists) else ""
            else:
                row_out[f"top{i+1}_object"] = ""
                row_out[f"top{i+1}_dist"] = ""

        neg1_obj = item.get("neg1_object")
        neg1_dist = item.get("neg1_dist")
        neg2_obj = item.get("neg2_object")
        neg2_dist = item.get("neg2_dist")
        row_out["neg1_object"] = neg1_obj if neg1_obj is not None else ""
        row_out["neg1_dist"] = neg1_dist if neg1_dist is not None else ""
        row_out["neg2_object"] = neg2_obj if neg2_obj is not None else ""
        row_out["neg2_dist"] = neg2_dist if neg2_dist is not None else ""
        csv_rows.append(row_out)

    out_csv = out_dir / "pfgw_top5" / "topk_summary.csv"
    ensure_dir(out_csv.parent)
    import csv
    fieldnames = [
        "query_graph",
        "true_object",
        "top1_object",
        "top1_dist",
        "top2_object",
        "top2_dist",
        "top3_object",
        "top3_dist",
        "top4_object",
        "top4_dist",
        "top5_object",
        "top5_dist",
        "neg1_object",
        "neg1_dist",
        "neg2_object",
        "neg2_dist",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--structure-train", type=int, default=0)
    ap.add_argument("--structure-test", type=int, default=30)
    ap.add_argument("--export-pfgw-top5", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--partial-mass", type=float, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    cfg = parse_config_effective(run_dir)
    k = infer_k_from_run(run_dir, cfg)

    graphs_root_candidates = [
        run_dir / "graphs",
        run_dir / "graphs" / f"k{k}" / "graphs",
    ]
    graphs_root = next((p for p in graphs_root_candidates if p.is_dir()), None)

    reps_root = run_dir / "reps" / "reps"
    mega_root = run_dir / "megagraphs" / "megagraphs"
    infer_json = run_dir / "inference" / "inference_result.json"

    if graphs_root is None:
        raise RuntimeError(
            "graphs root not found. Tried: "
            + ", ".join(str(p) for p in graphs_root_candidates)
        )
    if not reps_root.is_dir():
        raise RuntimeError(f"reps root not found: {reps_root}")
    if not mega_root.is_dir():
        raise RuntimeError(f"megagraphs root not found: {mega_root}")
    if not infer_json.is_file():
        raise RuntimeError(f"inference result not found: {infer_json}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else (run_dir / "analysis_export")
    ensure_dir(out_dir)

    infer_settings = read_json(infer_json).get("settings", {})
    alpha = float(args.alpha) if args.alpha is not None else float(infer_settings.get("alpha", cfg.get("infer", {}).get("alpha", 0.3)))
    partial_mass = (
        float(args.partial_mass)
        if args.partial_mass is not None
        else float(infer_settings.get("partial_mass", cfg.get("infer", {}).get("partial_mass", 0.7)))
    )

    export_structure_subset(graphs_root, out_dir, "train", int(args.structure_train), k)
    export_structure_subset(graphs_root, out_dir, "test", int(args.structure_test), k)
    export_representatives_and_anchors(run_dir, out_dir, k)
    export_object_graphs(run_dir, out_dir)
    export_pfgw_top5_from_inference(run_dir, out_dir, int(args.export_pfgw_top5), alpha, partial_mass)

    write_json(
        out_dir / "export_summary.json",
        {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "k": int(k),
            "structure_train": int(args.structure_train),
            "structure_test": int(args.structure_test),
            "export_pfgw_top5": int(args.export_pfgw_top5),
            "alpha": float(alpha),
            "partial_mass": float(partial_mass),
        },
    )
    print(f"[DONE] export: {out_dir}")


if __name__ == "__main__":
    main()
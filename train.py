#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def sha256_of_obj(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_command(cmd: List[str]) -> None:
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def make_stage_done(
    stage_name: str,
    status: str,
    started_at: float,
    finished_at: float,
    config_hash: str,
    outputs: Dict[str, Any],
    extras: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    obj = {
        "stage": stage_name,
        "status": status,
        "started_at_unix": started_at,
        "finished_at_unix": finished_at,
        "elapsed_sec": finished_at - started_at,
        "config_hash": config_hash,
        "outputs": outputs,
    }
    if extras:
        obj.update(extras)
    return obj


def is_stage_done(done_path: Path, expected_config_hash: str) -> bool:
    if not done_path.is_file():
        return False
    try:
        obj = read_json(done_path)
    except Exception:
        return False
    return obj.get("status") == "done" and obj.get("config_hash") == expected_config_hash


def resolve_cfg_path(config_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (config_dir / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    base_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(base_cfg, dict):
        raise RuntimeError("Config must be a YAML dict.")

    data_root = resolve_cfg_path(config_dir, str(base_cfg["data_root"]))
    script_root = resolve_cfg_path(config_dir, str(base_cfg["script_root"]))
    out_root = resolve_cfg_path(config_dir, str(base_cfg["out_root"]))
    ensure_dir(out_root)

    all_runs_summary: List[Dict[str, Any]] = []

    for seed in args.seeds:
        cfg = deepcopy(base_cfg)
        cfg.setdefault("graph", {})
        cfg["graph"]["seed"] = int(seed)

        k = int(cfg["graph"]["k"])
        run_name = f"seed_{seed}_k_{k}"
        run_dir = out_root / run_name
        checkpoints_dir = run_dir / "checkpoints"
        graphs_dir = run_dir / "graphs"
        reps_dir = run_dir / "reps"
        megagraphs_dir = run_dir / "megagraphs"
        inference_dir = run_dir / "inference"

        ensure_dir(run_dir)
        ensure_dir(checkpoints_dir)
        ensure_dir(graphs_dir)
        ensure_dir(reps_dir)
        ensure_dir(megagraphs_dir)
        ensure_dir(inference_dir)

        cfg_for_save = deepcopy(cfg)
        cfg_for_save["data_root"] = str(data_root)
        cfg_for_save["script_root"] = str(script_root)
        cfg_for_save["out_root"] = str(out_root)
        write_yaml(run_dir / "config_effective.yaml", cfg_for_save)

        graph_cfg_hash = sha256_of_obj({
            "stage": "A_graphs",
            "data_root": str(data_root),
            "graphs_dir": str(graphs_dir),
            "ann_root": str(cfg.get("ann_root", "")),
            "graph": cfg["graph"],
        })
        reps_cfg_hash = sha256_of_obj({
            "stage": "B_reps",
            "graphs_root": str(graphs_dir),
            "reps_dir": str(reps_dir),
            "rep": cfg["rep"],
        })
        mega_cfg_hash = sha256_of_obj({
            "stage": "C_megagraph",
            "reps_dir": str(reps_dir),
            "megagraphs_dir": str(megagraphs_dir),
            "megagraph": cfg["megagraph"],
            "rep_alpha": cfg["rep"]["alpha"],
        })
        infer_cfg_hash = sha256_of_obj({
            "stage": "D_infer",
            "query_graphs_root": str(graphs_dir / "test"),
            "megagraphs_dir": str(megagraphs_dir),
            "inference_dir": str(inference_dir),
            "infer": cfg["infer"],
            "rep_alpha": cfg["rep"]["alpha"],
        })

        stage_a_done = checkpoints_dir / "stage_A_graphs.done.json"
        stage_b_done = checkpoints_dir / "stage_B_reps.done.json"
        stage_c_done = checkpoints_dir / "stage_C_megagraph.done.json"
        stage_d_done = checkpoints_dir / "stage_D_infer.done.json"

        if not is_stage_done(stage_a_done, graph_cfg_hash):
            cmd_a = [
                "python",
                str(script_root / "tools" / "run_tokencut_graph_unified.py"),
                "--data-root",
                str(data_root),
                "--out-root",
                str(graphs_dir),
                "--tokencut-backbone",
                str(cfg["graph"]["tokencut_backbone"]),
                "--tok-resize",
                str(cfg["graph"]["tok_resize"]),
                "--tau",
                str(cfg["graph"]["tau"]),
                "--img-size",
                str(cfg["graph"]["img_size"]),
                "--k-list",
                str(k),
                "--seed",
                str(seed),
                "--fg-mode",
                str(cfg["graph"]["fg_mode"]),
                "--max-images",
                str(int(cfg["graph"].get("max_images", 0))),
                "--max-images-per-dir",
                str(int(cfg["graph"].get("max_images_per_dir", 0))),
            ]
            ann_root = str(cfg.get("ann_root", "")).strip()
            if ann_root:
                cmd_a += ["--ann-root", str(resolve_cfg_path(config_dir, ann_root))]

            t0 = time.perf_counter()
            run_command(cmd_a)
            t1 = time.perf_counter()

            graph_summary_path = graphs_dir / "graph_build_summary.json"
            graph_index_path = graphs_dir / "graph_index.json"

            write_json(
                stage_a_done,
                make_stage_done(
                    stage_name="A_graphs",
                    status="done",
                    started_at=t0,
                    finished_at=t1,
                    config_hash=graph_cfg_hash,
                    outputs={
                        "graphs_dir": str(graphs_dir),
                        "graph_summary_path": str(graph_summary_path),
                        "graph_index_path": str(graph_index_path),
                    },
                ),
            )

        if not is_stage_done(stage_b_done, reps_cfg_hash):
            cmd_b = [
                "python",
                str(script_root / "tools" / "eval_reps_barycenter_pascal3d.py"),
                "--graphs-root",
                str(graphs_dir),
                "--out-dir",
                str(reps_dir),
                "--fgw-alpha",
                str(cfg["rep"]["alpha"]),
            ]

            t0 = time.perf_counter()
            run_command(cmd_b)
            t1 = time.perf_counter()

            rep_summary_path = reps_dir / "rep_build_summary.json"

            write_json(
                stage_b_done,
                make_stage_done(
                    stage_name="B_reps",
                    status="done",
                    started_at=t0,
                    finished_at=t1,
                    config_hash=reps_cfg_hash,
                    outputs={
                        "reps_dir": str(reps_dir),
                        "rep_summary_path": str(rep_summary_path),
                    },
                ),
            )

        if not is_stage_done(stage_c_done, mega_cfg_hash):
            cmd_c = [
                "python",
                str(script_root / "tools" / "build_megagraphs_from_reps_eval.py"),
                "--rep-root",
                str(reps_dir),
                "--out-root",
                str(megagraphs_dir),
                "--alpha",
                str(cfg["megagraph"].get("alpha", cfg["rep"]["alpha"])),
                "--top-frac",
                str(cfg["megagraph"].get("top_frac", 0.2)),
            ]

            if cfg["megagraph"].get("threshold", None) is not None:
                cmd_c += ["--threshold", str(float(cfg["megagraph"]["threshold"]))]

            t0 = time.perf_counter()
            run_command(cmd_c)
            t1 = time.perf_counter()

            mega_summary_path = megagraphs_dir / "megagraph_build_summary.json"

            write_json(
                stage_c_done,
                make_stage_done(
                    stage_name="C_megagraph",
                    status="done",
                    started_at=t0,
                    finished_at=t1,
                    config_hash=mega_cfg_hash,
                    outputs={
                        "megagraphs_dir": str(megagraphs_dir),
                        "megagraph_summary_path": str(mega_summary_path),
                    },
                ),
            )

        if not is_stage_done(stage_d_done, infer_cfg_hash):
            cmd_d = [
                "python",
                str(script_root / "tools" / "infer_megagraph_classifier_eval.py"),
                "--query-graphs-root",
                str(graphs_dir / "test"),
                "--mega-root",
                str(megagraphs_dir),
                "--out-root",
                str(inference_dir),
                "--alpha",
                str(cfg["infer"].get("alpha", cfg["rep"]["alpha"])),
                "--topk",
                str(int(cfg["infer"].get("topk", 5))),
                "--max-queries",
                str(int(cfg["infer"].get("max_queries", 0))),
                "--partial-mass",
                str(cfg["infer"].get("partial_mass", 0.7)),
            ]

            t0 = time.perf_counter()
            run_command(cmd_d)
            t1 = time.perf_counter()

            inference_result_path = inference_dir / "inference_result.json"
            if not inference_result_path.is_file():
                raise RuntimeError(f"Stage D finished but inference_result.json was not created: {inference_result_path}")

            write_json(
                stage_d_done,
                make_stage_done(
                    stage_name="D_infer",
                    status="done",
                    started_at=t0,
                    finished_at=t1,
                    config_hash=infer_cfg_hash,
                    outputs={
                        "inference_dir": str(inference_dir),
                        "inference_result_path": str(inference_result_path),
                    },
                ),
            )

        inference_result_path = inference_dir / "inference_result.json"
        inference_result = read_json(inference_result_path) if inference_result_path.is_file() else {}

        run_summary = {
            "run_name": run_name,
            "seed": int(seed),
            "k": int(k),
            "paths": {
                "run_dir": str(run_dir),
                "checkpoints_dir": str(checkpoints_dir),
                "graphs_dir": str(graphs_dir),
                "reps_dir": str(reps_dir),
                "megagraphs_dir": str(megagraphs_dir),
                "inference_dir": str(inference_dir),
            },
            "checkpoints": {
                "stage_A_graphs": str(stage_a_done),
                "stage_B_reps": str(stage_b_done),
                "stage_C_megagraph": str(stage_c_done),
                "stage_D_infer": str(stage_d_done),
            },
            "results": {
                "top1": inference_result.get("summary", {}).get("top1"),
                "top5": inference_result.get("summary", {}).get("top5"),
                "eval_known": inference_result.get("summary", {}).get("eval_known"),
                "num_queries_total": inference_result.get("summary", {}).get("num_queries_total"),
                "num_skipped": inference_result.get("summary", {}).get("num_skipped"),
            },
            "timing": {
                "A_graph_sec": read_json(stage_a_done).get("elapsed_sec") if stage_a_done.is_file() else None,
                "B_rep_sec": read_json(stage_b_done).get("elapsed_sec") if stage_b_done.is_file() else None,
                "C_megagraph_sec": read_json(stage_c_done).get("elapsed_sec") if stage_c_done.is_file() else None,
                "D_infer_sec": read_json(stage_d_done).get("elapsed_sec") if stage_d_done.is_file() else None,
            },
        }

        timing_vals = [v for v in run_summary["timing"].values() if isinstance(v, (int, float))]
        run_summary["timing"]["total_sec"] = float(sum(timing_vals)) if timing_vals else None

        write_json(run_dir / "run_summary.json", run_summary)
        all_runs_summary.append(run_summary)
        print(
            f"[DONE] {run_name} "
            f"top1={run_summary['results']['top1']} "
            f"top5={run_summary['results']['top5']}"
        )

    write_json(out_root / "all_runs_summary.json", {"runs": all_runs_summary})


if __name__ == "__main__":
    main()
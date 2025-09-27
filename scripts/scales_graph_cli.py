#!/usr/bin/env python3
"""
CLI: Build a scale graph and analyze distances and components.

Graph options:
  - edge type 'flip': edges connect masks with Hamming distance 1 (add/remove a pitch)
  - edge type 'swap': edges connect masks with Hamming distance 2 while preserving k

Outputs (under --out-dir):
  - graph_nodes.csv (mask,k,degree,component_id)
  - graph_edges.csv (u,v)
  - degree_histogram.svg
  - component_sizes.svg
  - nearest_neighbors.csv (if --target is given)
  - shortest_path.csv (if --path-src and --path-dst are given)

Examples:
  python scripts/scales_graph_cli.py --out-dir out/graph --max-gap 4
  python scripts/scales_graph_cli.py --out-dir out/graph --edge-type swap --min-k 5 --max-k 7
  python scripts/scales_graph_cli.py --out-dir out/graph --target major --k 7
  python scripts/scales_graph_cli.py --out-dir out/graph --path-src major --path-dst minor --edge-type swap --k 7
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from music_analysis.plotting import apply_pub_style, save_svg
from music_analysis.theory import (
    N_PCS_DEFAULT,
    build_graph,
    graph_components,
    hamming,
    iter_scale_masks,
    parse_scale_spec,
    pcs_from_mask,
)


def main():
    ap = argparse.ArgumentParser(description="Scale graph and distance analysis")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=N_PCS_DEFAULT)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="If set, require exact k")
    ap.add_argument("--max-gap", type=int, default=None)
    ap.add_argument("--no-require-root", action="store_true")
    ap.add_argument("--edge-type", choices=["flip", "swap"], default="flip")
    ap.add_argument("--target", type=str, default=None, help="Find nearest neighbors to this scale (e.g., major, minor, mask:<int>, steps:2-2-1-...")
    ap.add_argument("--path-src", type=str, default=None)
    ap.add_argument("--path-dst", type=str, default=None)
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    require_root = not args.no_require_root

    min_k, max_k = args.min_k, args.max_k
    if args.k is not None:
        min_k = max_k = args.k

    masks = list(iter_scale_masks(n=args.n, require_root=require_root, min_k=min_k, max_k=max_k, max_gap=args.max_gap))
    g = build_graph(masks, n=args.n, edge_type=args.edge_type)

    # Degrees and components
    deg = {m: 0 for m in masks}
    for u, v in g.edges:
        deg[u] += 1
        deg[v] += 1
    comps = graph_components(g)
    comp_id = {}
    for i, comp in enumerate(comps):
        for m in comp:
            comp_id[m] = i

    nodes_df = pd.DataFrame({
        "mask": masks,
        "k": [bin(m).count("1") for m in masks],
        "degree": [deg[m] for m in masks],
        "component_id": [comp_id[m] for m in masks],
    })
    edges_df = pd.DataFrame(g.edges, columns=["u", "v"])
    nodes_df.to_csv(os.path.join(out, "graph_nodes.csv"), index=False)
    edges_df.to_csv(os.path.join(out, "graph_edges.csv"), index=False)

    # Plots
    apply_pub_style()
    plt.figure()
    plt.hist(nodes_df["degree"], bins=range(0, nodes_df["degree"].max() + 2))
    plt.xlabel("Node degree")
    plt.ylabel("Count")
    plt.title(f"Degree distribution ({args.edge_type} edges)")
    save_svg(os.path.join(out, "degree_histogram.svg"))

    comp_sizes = nodes_df.groupby("component_id").size().sort_values(ascending=False)
    plt.figure()
    plt.bar(range(len(comp_sizes)), comp_sizes.values)
    plt.xlabel("Component index (sorted)")
    plt.ylabel("Size")
    plt.title("Connected component sizes")
    save_svg(os.path.join(out, "component_sizes.svg"))

    # Nearest neighbors to target (if requested)
    if args.target is not None:
        tmask = parse_scale_spec(args.target, n=args.n)
        # restrict to same-k neighbors if --k is set; else allow all
        if args.k is not None:
            nodes = [m for m in masks if bin(m).count("1") == args.k]
        else:
            nodes = masks
        dists = [(m, hamming(m, tmask)) for m in nodes]
        mind = min(d for _, d in dists) if dists else 0
        nn = [(m, d) for m, d in dists if d == mind]
        nn_df = pd.DataFrame(nn, columns=["mask", "hamming_to_target"])
        nn_df.to_csv(os.path.join(out, "nearest_neighbors.csv"), index=False)

    # Shortest path if requested
    if args.path_src and args.path_dst:
        from music_analysis.theory import shortest_path

        src = parse_scale_spec(args.path_src, n=args.n)
        dst = parse_scale_spec(args.path_dst, n=args.n)
        if src in g.index and dst in g.index:
            path = shortest_path(g, src, dst)
            if path is not None:
                pd.DataFrame({"mask": path}).to_csv(os.path.join(out, "shortest_path.csv"), index=False)
        else:
            print("Source or destination not present under current constraints; skipping path.")

    print(f"Graph saved to {out} with {len(nodes_df)} nodes and {len(edges_df)} edges.")


if __name__ == "__main__":
    main()


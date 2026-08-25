import json
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
import features_store
import graph_builder as gb
from labels import load_split

OUT = "data/processed/graph_samples"


def export(n=24, n_nodes=30, edge_mode="knn", k=3):
    Path(OUT).mkdir(parents=True, exist_ok=True)
    split = load_split("test")
    picked, seen = [], {}
    for tid, info in split.items():
        g = info.get("genre", "")
        if not g or not features_store.has(tid):
            continue
        if seen.get(g, 0) >= 2:
            continue
        seen[g] = seen.get(g, 0) + 1
        picked.append((tid, info))
        if len(picked) >= n:
            break

    index = []
    for tid, info in picked:
        mel, chroma = gb.stitch(tid)
        x = gb.node_features(mel, chroma, n_nodes)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        ei, ea = gb.build_edges(x, edge_mode, k)
        seg = Data(x=torch.from_numpy(x), edge_index=ei, edge_attr=ea, num_nodes=n_nodes)
        seg.track_id = tid

        cx, cei, cea = gb.chord_graph(chroma)
        chord = Data(x=torch.from_numpy(cx), edge_index=cei, edge_attr=cea, num_nodes=cx.shape[0])
        chord.track_id = tid

        torch.save({"segment": seg, "chord": chord}, f"{OUT}/{tid}.pt")
        meta = {"segment": gb.graph_to_json(seg, tid, info),
                "chord": gb.graph_to_json(chord, tid, info)}
        json.dump(meta, open(f"{OUT}/{tid}.json", "w"), indent=2)
        index.append({"track_id": tid, "genre": info.get("genre"),
                      "segment_edges": int(ei.shape[1]), "chord_nodes": int(cx.shape[0])})

    json.dump(index, open(f"{OUT}/index.json", "w"), indent=2)
    print(f"exported {len(index)} example graphs (.pt + .json) to {OUT}")
    return index


if __name__ == "__main__":
    export(int(sys.argv[1]) if len(sys.argv) > 1 else 24)

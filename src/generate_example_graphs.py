import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from graph_builder import build_and_save_graphs, build_genre_mapping


def main(args):
    splits_dir = Path(args.splits_dir)

    for split_name, json_file in [
        ("training",   "training.json"),
        ("validation", "validation.json"),
        ("test",       "test.json"),
    ]:
        json_path = splits_dir / json_file
        if not json_path.exists():
            print(f"[SKIP] {json_path} not found.")
            continue

        with open(json_path) as f:
            split_dict = json.load(f)

        with open(splits_dir / "training.json") as f:
            train_dict = json.load(f)
        genre_to_idx, idx_to_genre = build_genre_mapping(train_dict)

        out_dir = Path(args.graph_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        max_graphs = None if args.all else args.n

        print(f"\n{'='*55}")
        print(f" Building {split_name} graphs  (max={max_graphs or 'ALL'})")
        print(f" Graph dir   : {out_dir}")
        print(f" Features dir: {args.processed_dir}")
        print(f"{'='*55}")

        saved = build_and_save_graphs(
            split_dict    = split_dict,
            genre_to_idx  = genre_to_idx,
            processed_dir = args.processed_dir,
            output_dir    = str(out_dir),
            sim_threshold = args.sim_threshold,
            max_graphs    = max_graphs,
            verbose       = True,
        )
        print(f" Saved {len(saved)} graphs for {split_name}.")

        print(f"\n--- Sample graphs (first 3) ---")
        for tid in saved[:3]:
            json_p = out_dir / f"{tid}.json"
            if json_p.exists():
                with open(json_p) as f:
                    meta = json.load(f)
                print(f"  {tid}: nodes={meta['num_nodes']}  edges={meta['num_edges']}  "
                      f"label={meta['label']}  genre={meta['genre']}")

        if not args.all and split_name == "training" and len(saved) >= 20:
            print(f"\n [OK] Submission requirement satisfied: {len(saved)} ≥ 20 example graphs.")
            break  


def parse_args():
    p = argparse.ArgumentParser(description="Generate example .pt/.json graph files")
    p.add_argument("--splits_dir",    default="data/splits")
    p.add_argument("--graph_dir",     default="data/processed/graphs")
    p.add_argument("--processed_dir", default="data/processed/audio_features")
    p.add_argument("--sim_threshold", type=float, default=0.7)
    p.add_argument("--n",    type=int, default=20, help="Number of graphs to build")
    p.add_argument("--all",  action="store_true",  help="Process entire dataset")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
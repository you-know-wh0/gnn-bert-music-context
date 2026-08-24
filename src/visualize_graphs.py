import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import graph_builder as gb
from labels import load_split

FEATURE_DIR = "data/processed/audio_features"
PLOTS = "plots"


def to_nx(edge_index, edge_attr, n_nodes):
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for (s, d), w in zip(edge_index.t().tolist(), edge_attr.tolist()):
        g.add_edge(s, d, weight=w)
    return g


def draw_segment_graph(ax, edge_index, edge_attr, n_nodes, title):
    g = to_nx(edge_index, edge_attr, n_nodes)
    ang = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False) - np.pi / 2
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(ang)}
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, ax=ax, width=[0.3 + 2.2 * w for w in weights],
                           edge_color=weights, edge_cmap=plt.cm.viridis, alpha=0.6,
                           arrows=False)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=110, node_color=range(n_nodes),
                           cmap=plt.cm.plasma, edgecolors="black", linewidths=0.5)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_chord_graph(ax, x, edge_index, edge_attr, title):
    nodes = np.argmax(x[:, :24], axis=1)
    labels = {i: gb.CHORD_NAMES[c] for i, c in enumerate(nodes)}
    g = to_nx(edge_index, edge_attr, len(nodes))
    pos = nx.circular_layout(g)
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    dur = x[:, -1]
    nx.draw_networkx_edges(g, pos, ax=ax, width=[0.3 + 3.0 * w for w in weights],
                           edge_color=weights, edge_cmap=plt.cm.magma, alpha=0.65,
                           arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=120 + 2400 * dur,
                           node_color="lightsteelblue", edgecolors="black", linewidths=0.6)
    nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=7)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def pick_examples(n=6):
    split = load_split("test")
    by_genre, out = {}, []
    for tid, info in split.items():
        g = info.get("genre", "")
        if g and g not in by_genre and os.path.exists(f"{FEATURE_DIR}/{tid}.npz"):
            by_genre[g] = tid
    for g, tid in sorted(by_genre.items())[:n]:
        out.append((tid, g))
    return out


def plot_segment_graphs(examples, n_nodes=30, edge_mode="knn", k=3, out="segment_graphs.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (tid, genre) in zip(axes.ravel(), examples):
        mel, chroma = gb.stitch(f"{FEATURE_DIR}/{tid}.npz")
        x = gb.node_features(mel, chroma, n_nodes)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        ei, ea = gb.build_edges(x, edge_mode, k)
        draw_segment_graph(ax, ei, ea, n_nodes, f"{genre} — {tid}\n{ei.shape[1]} edges")
    fig.suptitle(f"Segment graphs ({n_nodes} nodes, temporal + {edge_mode} similarity edges)\n"
                 "node colour = time position, edge colour/width = cosine similarity", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/{out}", dpi=150)
    plt.close(fig)


def plot_chord_graphs(examples, out="chord_graphs.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (tid, genre) in zip(axes.ravel(), examples):
        _, chroma = gb.stitch(f"{FEATURE_DIR}/{tid}.npz")
        x, ei, ea = gb.chord_graph(chroma)
        draw_chord_graph(ax, x, ei, ea, f"{genre} — {tid}\n{len(x)} chords, {ei.shape[1]} transitions")
    fig.suptitle("Chord-transition graphs (nodes = detected triads, size = duration share,\n"
                 "edge width = transition frequency)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/{out}", dpi=150)
    plt.close(fig)


def plot_similarity_matrices(examples, n_nodes=30, out="self_similarity.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (tid, genre) in zip(axes.ravel(), examples):
        mel, chroma = gb.stitch(f"{FEATURE_DIR}/{tid}.npz")
        x = gb.node_features(mel, chroma, n_nodes)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        im = ax.imshow(gb.cosine_sim(x), cmap="magma", vmin=-1, vmax=1)
        ax.set_title(f"{genre} — {tid}", fontsize=9)
        ax.set_xlabel("segment"); ax.set_ylabel("segment")
    fig.colorbar(im, ax=axes, shrink=0.7, label="cosine similarity")
    fig.suptitle(f"Segment self-similarity matrices ({n_nodes} segments per 30 s clip)", fontsize=11)
    fig.savefig(f"{PLOTS}/{out}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_graph_stats(n_tracks=800, n_nodes=30, out="graph_statistics.png"):
    split = load_split("training")
    ids = [t for t in list(split)[:n_tracks * 2] if os.path.exists(f"{FEATURE_DIR}/{t}.npz")][:n_tracks]
    modes = ["temporal", "knn", "threshold", "full"]
    edge_counts = {m: [] for m in modes}
    chord_nodes, chord_edges = [], []

    for tid in ids:
        mel, chroma = gb.stitch(f"{FEATURE_DIR}/{tid}.npz")
        x = gb.node_features(mel, chroma, n_nodes)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        for m in modes:
            edge_counts[m].append(gb.build_edges(x, m, 3, 0.7)[0].shape[1])
        cx, cei, _ = gb.chord_graph(chroma)
        chord_nodes.append(len(cx)); chord_edges.append(cei.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].boxplot([edge_counts[m] for m in modes], tick_labels=modes, showfliers=False)
    axes[0].set_title(f"Edges per segment graph ({n_nodes} nodes)")
    axes[0].set_ylabel("edge count"); axes[0].grid(alpha=0.3)

    axes[1].hist(chord_nodes, bins=range(2, 26), color="steelblue", edgecolor="black")
    axes[1].set_title("Distinct chords per track"); axes[1].set_xlabel("nodes"); axes[1].grid(alpha=0.3)

    axes[2].hist(chord_edges, bins=30, color="indianred", edgecolor="black")
    axes[2].set_title("Chord transitions per track"); axes[2].set_xlabel("edges"); axes[2].grid(alpha=0.3)

    fig.suptitle(f"Graph construction statistics over {len(ids)} training tracks", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/{out}", dpi=150)
    plt.close(fig)
    return {m: float(np.mean(v)) for m, v in edge_counts.items()} | {
        "chord_nodes_mean": float(np.mean(chord_nodes)),
        "chord_edges_mean": float(np.mean(chord_edges))}


if __name__ == "__main__":
    Path(PLOTS).mkdir(exist_ok=True)
    ex = pick_examples()
    print("examples:", ex)
    plot_segment_graphs(ex)
    plot_chord_graphs(ex)
    plot_similarity_matrices(ex)
    print("stats:", plot_graph_stats())
    print("plots written to", PLOTS)

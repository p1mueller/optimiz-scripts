import numpy as np

from dijkstra import GraphMatrix


def main():
    N = 6
    allowed_edges = None

    m = GraphMatrix(N)
    m[0, 1] = 45
    m[0, 2] = 150
    m[0, 3] = 970
    m[0, 4] = 690
    m[0, 5] = 590

    m[1, 2] = 220
    m[1, 3] = 130
    m[1, 4] = 390
    m[1, 5] = 810

    m[2, 3] = 15
    m[2, 4] = 930
    m[2, 5] = 490

    m[3, 4] = 800
    m[3, 5] = 130

    m[4, 5] = 10

    floyd_warhshall(m, allowed_edges)


def floyd_warhshall(graph, allowed_edges=None):
    N = len(graph)
    g = graph.m.copy()
    if allowed_edges is None:
        allowed_edges = np.arange(N)
    texts = []
    for k in allowed_edges:
        prop_dists = g[:, k : k + 1] + g[k : k + 1, :]
        mask = g > prop_dists
        g[mask] = prop_dists[mask]
        texts.append(f"Using intermediate edge {k}\n{g}")
    print("\n\n".join(texts))


if __name__ == "__main__":
    main()

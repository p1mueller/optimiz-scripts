import numpy as np


def main():
    N = 6
    start = 4
    m = GraphMatrix(N)
    m[0, 1] = 3
    m[0, 3] = 6
    m[0, 4] = 9

    m[1, 2] = 1
    m[1, 3] = 3
    m[1, 5] = 2

    m[2, 3] = 1
    m[2, 4] = 4
    m[2, 5] = 4

    m[3, 4] = 2

    m[4, 5] = 7

    prev_node, shortest_path, nodes_visited = dijkstra(m, start)
    for node, prev in enumerate(path(prev_node), 1):
        print(f"Previous node of node {node} is node {prev}")
    for end, dist in enumerate(shortest_path.astype(int), 1):
        print(f"Shortest distances from node {start + 1} to node {end} = {dist}")
    print(f"Order in which nodes were visited {path(nodes_visited[1:])}")
    print("Start point was removed.")


def path(path):
    return [int(v + 1) for v in path]


class GraphMatrix:
    def __init__(self, N):
        if type(N) is np.ndarray:
            m = N
        else:
            m = np.full((N, N), np.inf)
            i = np.arange(N)
            m[i, i] = 0
        self.m = m

    @classmethod
    def from_description(cls, graph):
        keys = sorted(graph.keys())
        N = len(keys)
        g = cls(N)
        for i, k in enumerate(keys):
            for node in graph[k]:
                j = keys.index(node)
                g[i, j] = 1
        return g

    def __copy__(self):
        return GraphMatrix(self.m)

    def __deepcopy__(self):
        return GraphMatrix(self.m.copy())

    def __getitem__(self, indices):
        return self.m[indices]

    def __setitem__(self, indices, value):
        i, j = indices
        self.m[i, j] = self.m[j, i] = value

    def __len__(self):
        return len(self.m)

    def copy(self):
        return self.__deepcopy__()

    def get_neighbors(self, node):
        nodes = np.arange(len(self))
        mask = self.m[node] != np.inf
        return nodes[mask]


def dijkstra(graph, start_node):
    N = len(graph)
    unvisited_nodes = np.arange(N).tolist()

    shortest_path = np.full(N, np.inf)
    shortest_path[start_node] = 0
    previous_nodes = np.full(N, np.inf)
    previous_nodes[start_node] = start_node

    nodes_visited = []

    while unvisited_nodes:
        # Find node with lowest score
        dists = shortest_path[unvisited_nodes]
        min_node = unvisited_nodes[np.argmin(dists)]

        # Update distances and previous nodes
        neighbors = graph.get_neighbors(min_node)
        tentative_values = shortest_path[min_node] + graph[min_node, neighbors]
        mask = tentative_values < shortest_path[neighbors]
        r = neighbors[mask]
        shortest_path[r] = tentative_values[mask]
        previous_nodes[r] = min_node

        unvisited_nodes.remove(min_node)
        nodes_visited.append(min_node)

    return previous_nodes, shortest_path, nodes_visited


if __name__ == "__main__":
    main()

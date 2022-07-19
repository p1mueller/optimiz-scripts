import numpy as np

from dijkstra import GraphMatrix


def main():
    start = "c"
    graph = {
        "a": ["b"],
        "b": ["a", "c", "f"],
        "c": ["b", "e", "f", "h"],
        "d": ["e", "g"],
        "e": ["c", "d", "i"],
        "f": ["b", "c", "h", "i"],
        "g": ["d", "i"],
        "h": ["c", "f", "i"],
        "i": ["e", "f", "g", "h"],
    }

    graph = {
        k.lower(): sorted([v.lower() for v in graph[k]]) for k in sorted(graph.keys())
    }
    print("BFS: " + " ".join(breadth_first_search(graph, start)))
    print("DFS: " + " ".join(depth_first_search(graph, start)))


def breadth_first_search(graph, start):
    explored = []
    queue = [start]
    # levels = {}  # this dict keeps track of levels
    # levels[start] = 0  # depth of start node is 0
    visited = [start]  # to avoid inserting the same node twice into the queue

    while queue:
        # pop shallowest node (first node) from queue
        node = queue.pop(0)
        explored.append(node)
        neighbours = graph[node]

        # add neighbours of node to queue
        for neighbour in neighbours:
            if neighbour not in visited:
                queue.append(neighbour)
                visited.append(neighbour)
    #             levels[neighbour] = levels[node] + 1
    # print(levels)
    return explored


def depth_first_search(graph, start):
    keys = sorted(graph.keys())
    N = len(keys)
    values = np.arange(N)

    node = keys.index(start)
    m = GraphMatrix.from_description(graph).m
    m[values, values] = np.inf

    stack = [node]
    visited = []
    while stack:
        node = stack[0]
        while True:
            mask = m[node] < np.inf
            if sum(mask) <= 0:
                break
            j = values[mask][0]
            stack.insert(0, j)
            m[:, node] = np.inf
            node = j
        node = stack.pop(0)
        m[:, node] = np.inf
        visited.append(node)

    return [keys[v] for v in visited]


if __name__ == "__main__":
    main()

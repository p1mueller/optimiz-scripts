# Optimization Scripts

Scripts for the exam of the MSE module "Optimization".

## Dependencies

For `branch_bound.py` and `simplex.py` LaTeX is needed. The author recommends using the `TeX Live` package.
Install Python dependencies with (Linux)

```
python3 -m pip install numpy sympy matplotlib pylatex intvalpy
```

## Description

Some of the files work only for some old exams.
You have to adapt them to your current exam.

| File                     | Description                                                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `branch_bound.py`        | Uses 'branch-and-bound' method to solve 0-1 knapsack problem. Creates tree image in `tex` folder of process.                                                                   |
| `brodyen.py`             | Applies Broydens method to find minimum of a function.                                                                                                                         |
| `dijkstra.py`            | Applies Dijkstras algorithm to a graph.                                                                                                                                        |
| `floyd-warshall.py`      | Applies Floyd-Warshall algorithm to a graph. With `allowed_edges` one can select which vertices the algorithm can use. If `None` all vertices can be used.                     |
| `gen_fitness.py`         | Calculates fitness of child in genetic algorithm (old exam)                                                                                                                    |
| `gen_knapsack.py`        | Calculate fitness of child in genetic algorithm to solve a 0-1 knapsack problem (old exam).                                                                                    |
| `graph_search.py`        | Implements 'breadth-first-search' and 'depth-first-search' algorithms for graphs.                                                                                              |
| `newton.py`              | Apply Newtons method to find minimum of a function.                                                                                                                            |
| `simplex.py`             | Apply simplex algorithm to a linear programming problem. Generates PDF file in `tex` folder with the graphical solution of the problem and the intermediate calculation steps. |
| `simulated_annealing.py` | Calculate simulated annealing probability to choose the proposition (old exam).                                                                                                |
| `successive_halving.py`  | Apply gradient-descent with successive halving with/without parabola fitting to find the minimum of a function.                                                                |

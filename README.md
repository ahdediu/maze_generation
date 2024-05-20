To be editted
We briefly introduce the notions of graphs and mazes following the notations from ...

A **graph** `g` is a tuple `g=(V,E)` where:

- `V` is a set of vertices,
- `E` is a set of edges such that `E={ {v1,v2} | v1, v2 are in V }`.

A **maze** `m` is a tuple `m=(V,E,b,e)` where:

- `(V,E)` is a graph,
- `b,e` are the start and end vertices of the maze respectively.

We focus on **connected mazes**, where there exists a path between any pair of vertices. For all `v1, v2` in `V`, there exists a path in the graph from `v1` to `v2`.
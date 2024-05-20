To be editted
We briefly introduce the notions of graphs and mazes following the notations from ...

A \emph{graph} $g$ is a tuple $g=(V,E)$ where:
\begin{itemize}
  \item[] $V$ is a set of vertices,
  \item[] $E=\{\{v_1,v_2\} \,|\, v_1, v_2 \in V\}$ is a set of edges.
\end{itemize}

A \emph{maze} $m$ is a tuple $m=(V,E,b,e)$ where:
\begin{itemize}
  \item[] $(V,E)$ is a graph,
  \item[] $b,e$ are the start and end vertices of the maze respectivelly.
\end{itemize}

We focus on \emph{connected mazes}, where there exists a path between any pair of vertices. For all $v_1, v_2 \in V$, there exists a path in the graph from $v_1$ to $v_2$.
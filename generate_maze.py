import collections
import random
from symtable import Symbol
from typing import Set, Tuple, List, FrozenSet, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import patches
from pyformlang.finite_automaton import DeterministicFiniteAutomaton, State
import os
import networkx as nx


class Maze:
    def __init__(self, m: int, n: int, start: Tuple = (0, 0), end: tuple = None) -> None:
        self.m: int = m
        self.n: int = n
        self.start: int = start
        self.end: int = end if end is not None else (m - 1, n - 1)
        self.vertices = set()
        self.edges = set()
        self.unvisited_vertices = {(i, j) for i in range(m) for j in range(n)}
        # Start with a single random vertex
        self.random_walk_until_meets_maze(meet_itself=True)
        while len(self.vertices) < self.m * self.n:
            self.random_walk_until_meets_maze(meet_itself=False)

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertices)
        self.graph.add_edges_from(self.edges)

        self.remove_extra_edges()
        #peek = iter(self.edges) then next(peek) #
        self.solution_path: Set[FrozenSet[Tuple[int, int]]] =  self.find_path()

        """
        remove extra edges is tricky
        When find_path is called, it determines a temporary path from the start to the current vertex. 
        But only removing an edge not on this temporary path doesn't guarantee that 
        the vertex is still connected to the start point. 
        The edge you're removing might be part of a different path that's 
        currently the only way to reach another vertex.
        """
        self.vertex_labels = {}  # New member to store vertex labels
        for vertex in self.vertices:
            self.vertex_labels[vertex] = '+' if random.choice([True, False]) else '-'


    def is_connected(self) -> bool:
        return nx.is_connected(self.graph)
    def remove_extra_edges(self) -> None:
        for vertex in self.vertices:
            vertex_edges = [e for e in self.edges if vertex in e]
            if len(vertex_edges) > 3:
                for edge_to_try in vertex_edges:
                    # Temporarily remove the edge from both self.edges and self.graph
                    self.edges.remove(edge_to_try)
                    self.graph.remove_edge(*edge_to_try)
                    if not self.is_connected():
                        self.edges.add(edge_to_try)
                        self.graph.add_edge(*edge_to_try)
                    else:
                        break

    # function to find a path, say depth-first search
# from top left to bottom right of the maze

# get the path
#     def find_path(self, end_path: Optional[Tuple[int, int]] = None) -> Optional[Set[FrozenSet[Tuple[int, int]]]]:
#         """
#         Perform a breadth-first search to find a path from start to end.
#
#         :param end_path: The end point for the path calculation.
#         :return: Set of frozensets. Each frozenset represents an edge,
#                  expressed as a pair of vertices (where each vertex is a tuple of coordinates).
#                  None if no such path exists.
#         """
#         if end_path is None:
#             end_path = self.end
#
#         visited = set()  # A set to store visited vertices.
#         queue = collections.deque([[self.start]])  # Initialize the queue with the start point.
#
#         while queue:  # Continue until all paths are exhausted.
#             path = queue.popleft()  # Pop the next path from the queue.
#             vertex = path[-1]  # Get the last vertex from the path.
#
#             if vertex in visited:  # If the vertex has been visited, skip to next iteration
#                 continue
#
#             visited.add(vertex)  # Mark the current vertex as visited
#
#             for adjacent in self.adjacent_vertices(vertex):  # iterate over adjacent vertices
#                 new_path = list(path)  # Create a new path from the old one
#                 new_path.append(adjacent)  # Add the adjacent vertex to the new path
#
#                 if adjacent == end_path:  # Found a path
#                     return {frozenset({new_path[i - 1], new_path[i]}) for i in range(1, len(new_path))}
#
#                 queue.append(new_path)  # Add the new path to the queue
#
#         return None  # No path was found
    def find_path(self) -> Set[FrozenSet[tuple]]:
        try:
            # Use NetworkX's shortest path function
            path = nx.shortest_path(self.graph, self.start, self.end)
            # Convert the returned node path into edge path, and use frozenset for edges
            edge_path = {frozenset([path[i - 1], path[i]]) for i in range(1, len(path))}
            return edge_path
        except nx.NetworkXNoPath:
            print("No path between start and end.")
            return set()

    def add(self, vertices: List[Tuple[int, int]], edges: Set[FrozenSet[Tuple[int, int]]]) -> None:
        self.vertices.update(vertices)
        self.edges.update(edges)
        self.unvisited_vertices.difference_update(vertices)


    def edges_of_vertex(self, vertex: Tuple[int, int]) -> Set[Set[Tuple[int, int]]]:
        """Returns a set of valid edges for a given vertex within the grid."""
        i, j = vertex
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # right, left, down, up
        edges = set()

        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < self.m and 0 <= nj < self.n:  # check if the neighbour is within the grid
                edge = {vertex, (ni, nj)}
                edges.add(frozenset(edge))  # we use frozenset to be able to add sets into a set

        return edges


    def random_walk_until_meets_maze(self, meet_itself: bool) -> None:
        # collateral effect, updates the edges and the vertices of the maze itself.
        if not self.unvisited_vertices:
            # Handle if there are no unvisited vertices, e.g. return, or raise an exception.
            raise Exception("No more unvisited vertices!")

        current_vertex = random.choice(list(self.unvisited_vertices))

        walk_vertices = [current_vertex]
        walk_edges = set()

        while True:
            edges_of_current_vertex = {e for e in self.edges if
                                       current_vertex in e}  # existing edges of the current vertex
            if len(edges_of_current_vertex) >= 3:  # if current vertex already has 3 edges, break the walk
                break
            adjacent_edges = self.edges_of_vertex(current_vertex)
            current_edge = random.choice(list(adjacent_edges))
            next_vertex = next(iter(current_edge - {current_vertex}))
            walk_vertices.append(next_vertex)
            walk_edges.add(current_edge)
            if meet_itself and next_vertex in walk_vertices:
                self.add(walk_vertices, walk_edges)
                return
            elif next_vertex in self.vertices:
                self.add(walk_vertices, walk_edges)
                return
            current_vertex = next_vertex
        # this code is never reached
        return


    def draw_maze(self) -> None:
        """
          This method visualizes the maze.
          It uses Matplotlib to draw the maze's paths.
          Red lines indicate the solution of the maze.
          The maze is drawn in a separate pop-up window.
          """
        fig, ax = plt.subplots()

        ax.axis('off')  # turning off the axis
        # getting x and y coordinates of all points in the maze

        x_values, y_values = zip(*[point for edge in self.edges for point in edge])
        ax.set_xlim(min(x_values) - 1, max(x_values) + 1)
        ax.set_ylim(min(y_values) - 1, max(y_values) + 1)

        ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))
        ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))

        ax.grid(which='major', color='k')
        # drawing the edges of the maze
        for edge in self.edges:
            x1, y1 = list(edge)[0]
            x2, y2 = list(edge)[1]
            if edge in self.solution_path:
                ax.plot([x1, x2], [y1, y2], color='r', linewidth=2)
            else:
                ax.plot([x1, x2], [y1, y2], color='k', linewidth=1)
        radius = 0.35
        for vertex, label in self.vertex_labels.items():
            x, y = vertex
            if label == '+':
                # ax.text(x, y, label,color="blue", fontsize=12)
                circ = patches.Circle(vertex, radius/3, edgecolor='blue', facecolor='blue')
                ax.add_patch(circ)
            if label == '-':
                #label = "\u25CF"  # White Circle (which appears as opaque)
                #ax.text(x, y, "\u3280", ha='center', va='center')
                circ = patches.Circle(vertex, radius, edgecolor='blue', facecolor='none')

                # Add the circle to the plot
                ax.add_patch(circ)

        # space for lables
        # for (x1, y1), (x2, y2) in maze:
        #     ax.text(x1, y1, str(x1 + y1*3), color="blue", fontsize=9,ha='center', va='center' )
        #     ax.text(x2, y2, str(x2  + y2*3), color="blue", fontsize=9,ha='center', va='center' )
        plt.show()  # display the plot


def create_GAP_transition_table(maze: Set[Tuple[int, int]], m: int, n: int) -> list:
    transition_table = [[i for i in range(m * n)] for _ in range(4)]  # Default transition is self

    for edge in maze:
        u, v = edge
        ux, uy = u  # unpack the tuple coordinates
        vx, vy = v

        # Calculate state numbers
        state_u = ux + uy * m
        state_v = vx + vy * m

        if ux == vx:  # the edge is vertical    uy
            if uy > vy:  # the edge goes N-S    vy
                transition_table[2][state_u] = state_v  # u transitions to v on 's'
                transition_table[0][state_v] = state_u  # v transitions to u on 'n'
            else:  # the edge goes S-N
                transition_table[0][state_u] = state_v  # u transitions to v on 'n'
                transition_table[2][state_v] = state_u  # v transitions to u on 's'
        else:  # the edge is horizontal
            if ux > vx:  # the edge goes vx ux
                transition_table[3][state_u] = state_v  # u transitions to v on 'w'
                transition_table[1][state_v] = state_u  # v transitions to u on 'e'
            else:  # the edge goes ux vx
                transition_table[1][state_u] = state_v  # u transitions to v on 'e'
                transition_table[3][state_v] = state_u  # v transitions to u on 'w'

    return transition_table


def create_dfa(maze: Maze) -> DeterministicFiniteAutomaton:
    transition_table: list = create_GAP_transition_table(maze.edges, maze.m, maze.n)
    dfa = DeterministicFiniteAutomaton()

    direction_to_symbol = {
        0: Symbol('n'),
        1: Symbol('e'),
        2: Symbol('s'),
        3: Symbol('w')
    }

    num_states = len(transition_table[0])

    for direction in range(4):  # For each direction
        for state in range(num_states):  # For each state

            # Get mapped Symbol
            symbol = direction_to_symbol[direction]

            # Get next state from transition table
            next_state = transition_table[direction][state]

            # Define state and next_state
            s = State(str(state))
            s_next = State(str(next_state))

            # Add the transition to the DFA
            dfa.add_transition(s, symbol, s_next)

    dfa.add_start_state(State(maze.start))
    dfa.add_final_state(State(maze.end))
    return dfa


def dfa_is_minimal(dfa: DeterministicFiniteAutomaton):
    from pyformlang import finite_automaton as fa
    minimal_dfa = dfa.minimize()
    return dfa == minimal_dfa


m: int = 25  # horizontal grid size
n: int = 20  # vertical grid size
maze = Maze(m, n)
maze.draw_maze()
# t=create_GAP_transition_table(maze.edges, m, n)
# print(t)
# Initialize the automaton (replace this with your current automaton)
dfa = DeterministicFiniteAutomaton()
# ... (define your dfa here)
# dfa.tr
# Check for minimality
is_minimal = dfa_is_minimal(dfa)
print(f'The automaton is {"minimal" if is_minimal else "not minimal"}')
# Save the result to a text file
# Be sure to use the path where you want to save the results
# with open('path/to/your/minimality_result.txt', 'w') as file:
#    file.write(f'The automaton is {"minimal" if is_minimal else "not minimal"}.\n')

# print(f'Minimality result saved to text file.')

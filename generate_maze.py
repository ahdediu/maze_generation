import collections
import itertools
import random
from symtable import Symbol
from typing import Set, Tuple, List, FrozenSet, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyformlang.finite_automaton import DeterministicFiniteAutomaton, State
import os
class Maze:
    def __init__(self, m: int, n: int,start:Tuple=(0,0), end: tuple = None) -> None:
        self.m = m
        self.n = n
        self.start = start
        self.end = end if end is not None else (m - 1, n - 1)
        self.vertices = set()
        self.edges = set()
        self.unvisited_vertices = {(i, j) for i in range(m) for j in range(n)}
        # Start with a single random vertex
        self.random_walk_until_meets_maze(meet_itself=True)
        while len(self.vertices) < self.m * self.n:
            self.random_walk_until_meets_maze(meet_itself=False)
        self.remove_extra_edges()
    def remove_extra_edges(self):

        # function to find a path, say depth-first search
        # from top left to bottom right of the maze

        # get the path
        path = self.find_path()

        # iterate through all vertices, if its not in path
        # and it has more than 3 edges, remove one at random
        for vertex in self.vertices:
            if vertex not in path:
                edges = [e for e in self.edges if vertex in e]
                if len(edges) > 3:
                    self.edges.remove(random.choice(edges))

        # for vertices on the path with more than 3 edges
        # remove an edge which is not on the path.
        for vertex in path:
            edges = [e for e in self.edges if vertex in e]
            if len(edges) > 3:
                for edge in edges:
                    if not any(e in path for e in edge):
                        self.edges.remove(edge)
                        break

        return

    def adjacent_vertices(self, vertex):
            return [v for edge in self.edges for v in edge
                    if v != vertex and edge not in self.edges_of_vertex(vertex)]

    def find_path(self) -> Optional[List[Tuple[int, int]]]:
            """
            Perform a breadth-first search to find a path from start to end.

            :return: List of tuples representing the path from start_point to end_point, or
                     None if no such path exists.
            """

            visited = set()  # A set to store visited vertices.
            queue = collections.deque([[self.start]])  # Initialize the queue with the start point.

            while queue:  # Continue until all paths are exhausted.
                path = queue.popleft()  # Pop the next path from the queue.
                vertex = path[-1]  # Get the last vertex from the path.

                # If this vertex is the end point, we've found a path!
                if vertex == self.end:
                    return path

                elif vertex not in visited:  # If the vertex has not been visited yet,
                    for adjacent in self.adjacent_vertices(vertex):  # iterate over adjacent vertices
                        new_path = list(path)  # Create a new path from the old one
                        new_path.append(adjacent)  # Add the adjacent vertex to the new path
                        queue.append(new_path)  # Add the new path to the queue

                    visited.add(vertex)  # Mark the current vertex as visited

            return None  # No path was found

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


    def random_walk_until_meets_maze(self, meet_itself:bool)-> None:
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



def create_GAP_transition_table(maze: Set[Tuple[int, int]], m: int, n: int) -> list:
    transition_table = [[i for i in range(m * n)] for _ in range(4)]  # Default transition is self

    for edge in maze:
        u, v = edge
        ux, uy = u  # unpack the tuple coordinates
        vx, vy = v

        # Calculate state numbers
        state_u = ux  + uy*m
        state_v = vx  + vy*m

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




def draw_maze(maze):
    fig, ax = plt.subplots()

    ax.axis('off')

    x_values, y_values = zip(*[point for edge in maze for point in edge])
    ax.set_xlim(min(x_values) - 1, max(x_values) + 1)
    ax.set_ylim(min(y_values) - 1, max(y_values) + 1)

    ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))
    ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))

    ax.grid(which='major', color='k')

    for (x1, y1), (x2, y2) in maze:
        ax.plot([x1, x2], [y1, y2], color='r', linewidth=1)

    # for (x1, y1), (x2, y2) in maze:
    #     ax.text(x1, y1, str(x1 + y1*3), color="blue", fontsize=9,ha='center', va='center' )
    #     ax.text(x2, y2, str(x2  + y2*3), color="blue", fontsize=9,ha='center', va='center' )
    plt.show()
def create_dfa(maze:Maze) -> DeterministicFiniteAutomaton:
    transition_table: list=create_GAP_transition_table(maze.edges, maze.m, maze.n)
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



m:int=25 # horizontal grid size
n:int=20 # vertical grid size
maze = Maze(m, n)
draw_maze(maze.edges)
#t=create_GAP_transition_table(maze.edges, m, n)
#print(t)
# Initialize the automaton (replace this with your current automaton)
dfa = DeterministicFiniteAutomaton()
# ... (define your dfa here)
#dfa.tr
# Check for minimality
is_minimal = dfa_is_minimal(dfa)
print(f'The automaton is {"minimal" if is_minimal else "not minimal"}')
# Save the result to a text file
# Be sure to use the path where you want to save the results
#with open('path/to/your/minimality_result.txt', 'w') as file:
#    file.write(f'The automaton is {"minimal" if is_minimal else "not minimal"}.\n')

#print(f'Minimality result saved to text file.')


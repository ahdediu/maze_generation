
import random

from typing import Set, Tuple, List, FrozenSet, Optional


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import patches
import os
import networkx as nx

import json
import datetime

class Maze:
    def __init__(self, m: int, n: int, start: Tuple = (0, 0), end: tuple = None) -> None:
        self.m: int = m
        self.n: int = n
        self.vertex_labels = {}  # New member to store vertex labels
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
        self.solution_vertex_path:List[Tuple[int, int]]=self.find_solution()
        self.solution_path: Set[FrozenSet[Tuple[int, int]]] =  self.convert_path_to_edges(self.solution_vertex_path)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.filename = f"maze_{m}_by_{n}_ortho_{self.timestamp}."

        """
        remove extra edges is tricky
        When find_path is called, it determines a temporary path from the start to the current vertex. 
        But only removing an edge not on this temporary path doesn't guarantee that 
        the vertex is still connected to the start point. 
        The edge you're removing might be part of a different path that's 
        currently the only way to reach another vertex.
        """

    def __getstate__(self):
        # Return a dict of attributes you want to pickle.
        return {
            'width': self.m,
            'height': self.n,
            'start': self.start,
            'end': self.end,
            'edges': [[vertex for vertex in edge] for edge in self.edges],
            'solution_path': self.solution_vertex_path,
            'labels': ''.join([self.vertex_labels[(i, j)] for i in range(self.m) for j in range(self.n)]),
            'labels_comment': 'all labels for vertices [0,0], [0,1], ... [m-1,n-1]',

        }

    def save(self, dir_name: str="../data") -> None:
        # Creating a filename using the timestamp

        data = self.__getstate__()
        json_str = json.dumps(data)
        formatted_json_str = json_str.replace(', "', ',\n "')
        # Joining provided directory path and filename
        file_path = os.path.join(dir_name, self.filename+"json")

        # Save dict as a json in the provided location
        with open(file_path, 'w') as f:
            # json_str = jsonpickle.encode(self, indent=4)
            # f.write(json_str)
            f.write(formatted_json_str)
        self.draw_maze(dir_name+"//"+self.filename+"png")


    def set_random_labels(self):
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

    def find_solution(self)-> List[Tuple[int, int]]:
        path:List[Tuple[int, int]] = []  # Initialize path as an empty list
        try:
            # Use NetworkX's shortest path function
            path = nx.shortest_path(self.graph, self.start, self.end)
        except nx.NetworkXNoPath:
            print("No path between start and end.")
        return path

    def convert_path_to_edges(self,path: List[Tuple[int, int]]) -> Set[FrozenSet[tuple]]:
            # Convert the returned node path into edge path, and use frozenset for edges
            edge_path = {frozenset([path[i - 1], path[i]]) for i in range(1, len(path))}
            return edge_path


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


    def draw_maze(self,save:str=None) -> None:
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
        if self.vertex_labels :
            radius = 0.3
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

        # space for state numbers
        # for x in range(self.m):
        #     for y in range(self.n):
        #         ax.text(x, y, str(x + y*self.m), color="k", fontsize=5,ha='center', va='center' )
        if save:
            plt.savefig(save)
            plt.close(fig)
        else:
            plt.show()  # display the plot




m: int = 25  # horizontal grid size
n: int = 20  # vertical grid size
maze = Maze(m, n)
maze.set_random_labels()
maze.draw_maze()

maze.save()

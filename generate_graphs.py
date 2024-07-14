from collections import deque
from typing import List, Tuple

from matplotlib import pyplot as plt

import random
import networkx as nx

''' Initial walls: mxnx2 (L shape walls for each cell)+m+n 
	to close the top and the right borders. 
	Now each edge can be considered as visiting one cell minus the last one 
	that visits 2 cells. so mxm-1 edges.  '''


class Maze:
	def __init__(self, m, n, wall_chance=0.5):
		self.m = m
		self.n = n
		self.wall_chance = wall_chance
		self.graph = nx.Graph()
		self.visited = [[False] * n for _ in range(m)]
		self.visited[0][0] = True
		self.graph.add_node((0, 0))
		self.make_maze()

	directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
	def simulate_random_walk(self, l):
		# Choose a random starting position
		start = (random.randint(0, self.m - 1), random.randint(0, self.n - 1))

		visited_rooms = set()
		current_position = start

		for _ in range(l):
			visited_rooms.add(current_position)

			# Choose a random direction to move in.
			# If we hit a wall, we stay in the same room.
			direction = random.choice(self.directions)
			new_position = (current_position[0] + direction[0], current_position[1] + direction[1])

			if new_position in self.graph[current_position]:
				# If there's no wall in this direction, move to the new room.
				current_position = new_position

		return len(visited_rooms)

	def run_simulation(self, n_simulations, l):

		visited_rooms = []
		for _ in range(n_simulations):
			visited_rooms.append(maze.simulate_random_walk(l))

		return visited_rooms

	def make_maze(self):
		self.random_walk_maze((0, 0))

	def random_walk_maze(self, start):
		dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]  # directions to get the four neighbors
		stack = [start]
		visited = {(start): True}

		while stack:
			current = stack[-1]

			# Get list of unvisited neighbors
			neighbors = [(current[0] + dx[i], current[1] + dy[i]) for i in range(4) if
						 0 <= current[0] + dx[i] < self.m and
						 0 <= current[1] + dy[i] < self.n and
						 (current[0] + dx[i], current[1] + dy[i]) not in visited]

			if neighbors:
				# Choose a random neighboring cell and mark as visited
				next_cell = random.choice(neighbors)
				visited[(next_cell)] = True
				stack.append(next_cell)

				# remove the wall between cells
				self.graph.add_edge(current, next_cell)
			else:
				# Backtrack
				stack.pop()

	def find_solution(self) -> List[Tuple[int, int]]:
		path: List[Tuple[int, int]] = []  # Initialize path as an empty list
		try:
			# Use NetworkX's shortest path function
			path = nx.shortest_path(self.graph, self.start, self.end)
		except nx.NetworkXNoPath:
			print("No path between start and end.")
		return path
	def _dfs(self, node):
		directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
		random.shuffle(directions)
		x, y = node
		self.visited[x][y] = True
		for dx, dy in directions:
			new_x, new_y = x + dx, y + dy
			if 0 <= new_x < self.m and 0 <= new_y < self.n and not self.visited[new_x][new_y]:
				self.graph.add_edge(node, (new_x, new_y))
				self._dfs((new_x, new_y))

	# def plot_maze(self):
	# 	pos = {(x, y): (y, -x) for x in range(self.m) for y in range(self.n)}
	# 	nx.draw(self.graph, pos=pos, with_labels=True, node_color='lightblue', font_color='black')
	def find_solution(self,start,end) -> List[Tuple[int, int]]:
		path: List[Tuple[int, int]] = []  # Initialize path as an empty list
		try:
			# Use NetworkX's shortest path function
			path = nx.shortest_path(self.graph, start, end)
		except nx.NetworkXNoPath:
			print("No path between start and end.")
		return path

	def plot_maze(self, solution_path=None):
		pos = {(x, y): (y, -x) for x in range(self.m) for y in range(self.n)}  # positions for all nodes

		# edges
		nx.draw_networkx_edges(self.graph, pos, width=0.5)

		# nodes (drawn as small dots)
		nx.draw_networkx_nodes(self.graph, pos, node_size=4, node_color='blue')

		if solution_path:
			# We change solution_path from nodes list to edge list
			edge_list = [(solution_path[i], solution_path[i + 1]) for i in range(len(solution_path) - 1)]

			# Draw solution edges with color red
			nx.draw_networkx_edges(self.graph, pos, edgelist=edge_list, edge_color='red', width=1)


		plt.axis('off')
		plt.savefig("graph.png")
		plt.show()


class automaton_graph:
	def __init__(self, vertices: int, alphabet_size: int, maze: bool = True):
		"""
		Args:
		  m: vertices in the graph.
		  k: alphabet_size, Number of edges per vertex.
		  maze: If True, the graph is maze. For maze for each edge direction there is a complementary one
		  	practically we have 2*k-1 alphabet size.
		  	however, k stays the same only the edges labels will be greater than k
		"""

		self.m = vertices
		self.k = alphabet_size
		self.maze = maze

		if self.maze:
			self.graph = nx.Graph()  # undirected graph
			for i in range(self.m):
				self.graph.add_node(i)
			t = {}
			for j in range(self.k):
				t[j] = random.sample(range(self.m), self.m)
			for i in range(self.m):
				for j in range(self.k):
					self.graph.add_edge(i, t[j][i], label=j)


		else:
			self.graph = nx.DiGraph()
			self.vertex_level = {}
			for i in range(self.m):
				# self.graph.add_node(i)
				self.vertex_level[i] = self.m + 1

			# edges later as for an edge we need 2 nodes.
			#  self.graph.add_edge(u,v)

			# no need to keep the edges on levels
			self.vertex_level[0] = 0
			self.graph.add_node(0)
			level = 1
			current_level = [0]
			while True:
				next_level = []
				for v in current_level:
					for i in range(self.k):
						random_vertex = random.choice(range(self.m))
						# self.graph.add_edge(v, random_vertex,label=i)
						self.graph.add_edge(v, random_vertex, label=i)
						if self.vertex_level[random_vertex] > level:
							self.graph.add_node(random_vertex)
							self.vertex_level[random_vertex] = level
							next_level.append(random_vertex)
				if next_level == []:
					for edge in self.graph.edges(data=True):
						print(edge)
					break
				else:
					current_level = next_level
					level += 1
			if self.graph.number_of_nodes() < self.m:
				print(f"Only {self.graph.number_of_nodes()} nodes are connected. Max level: {level}")

	def draw_graph(self):
		pos = nx.fruchterman_reingold_layout(
			self.graph)  # options: kamada_kawai_layout or nx.fruchterman_reingold_layout(self.graph)
		# pos = nx.spring_layout(self.graph, k=1.2)
		# nodes
		nx.draw_networkx_nodes(self.graph, pos)

		# edges
		nx.draw_networkx_edges(self.graph, pos)

		# labels for nodes
		nx.draw_networkx_labels(self.graph, pos)

		# labels for edges
		if self.maze:
			edge_labels = {(u, v): d['label'] for u, v, d in self.graph.edges(data=True) if d['label'] < self.k}
			nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10, font_color='red')

		plt.axis('off')
		plt.savefig("graph.png")
		plt.show()


# g =automaton_graph(50, 2, maze=True)
# g.draw_graph()
maze = Maze(25, 20)
# print(maze.find_solution((0, 0), (6, 8)))
maze.plot_maze(maze.find_solution((0, 0), (maze.m-1, maze.n-1)))
plt.show()
# visited_rooms = maze.run_simulation(n_simulations=100, l=5)

# Compute the average number of visited rooms
# average_visited_rooms = sum(visited_rooms) / len(visited_rooms)
# print(f"Average number of rooms visited: {average_visited_rooms}")
